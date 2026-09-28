# `alignas(128)` on a `__grid_constant__` TMA descriptor: C2719 on MSVC, and the fix

Research into the C2719 failure on the new upstream fp8/nvfp4/bf16 TMA linear kernels, into the
canonical fix, and into whether any of the alternatives cost performance. Verified on 2026-09-28
against CUDA 13.3 (`V13.3.73`), MSVC 14.51 (VS 18 Build Tools), and an RTX 5090 (`sm_120a`), with
the offending declarations read from `upstream/dev` (`7f6aafed`).

## Recommendation

**Delete the `alignas(128)` and change nothing else.** The by-value `const __grid_constant__`
parameter is the form NVIDIA documents as *recommended* for a tensor map, and it is already the
form this port uses. The `alignas(128)` is the entire bug: it is a host-ABI constraint that has no
device-side meaning, and it is the only thing that differs between the working port and the
failing upstream code.

```diff
--- a/src/ops/linear/fp8/fp8_a8_tma_mma.cuh
+++ b/src/ops/linear/fp8/fp8_a8_tma_mma.cuh
-struct alignas(128) Fp8TmaDescriptors {
+// No alignas override: CUtensorMap already declares alignas(TENSOR_MAP_ALIGN), which is 64
+// under _MSC_VER and 128 elsewhere. Overriding it to 128 makes nvcc's host stub declare a
+// 128-byte-aligned formal parameter, which MSVC's x64 convention cannot honour (C2719).
+struct Fp8TmaDescriptors {
     CUtensorMap activation;
     CUtensorMap weight;
 };
```

The identical one-line change applies to `Nvfp4A4TmaDescriptors`
(`src/ops/linear/nvfp4/nvfp4_a4_tma.cuh:27`) and `Bf16TmaDescriptors`
(`src/ops/linear/bf16/bf16_a16_tma_mma.cuh:20`), which carry `alignas(128)` on `upstream/dev` and
must not be reintroduced by a merge. The port already made exactly this change in `1218d574`
("fix(nvfp4): pass the W4A4 TMA descriptor by value so it survives graph capture") and recorded
the reason in comments at those two sites; this note establishes it from primary sources.

## Verdicts at a glance

| Question | Verdict | Basis |
|---|---|---|
| Documented nvcc limitation or MSVC bug? | **MSVC cannot represent it** — a hard ABI limit, not a front-end bug | x64 calling convention documents 16 bytes as the ceiling; measured: MSVC accepts `alignas(64)`, rejects 128 |
| Canonical fix | **Remove the `alignas` override; keep by-value `__grid_constant__`** | Programming Guide §4.12.2.2 names this the *recommended* approach and names the other two as alternatives |
| Does `const T&` work? | **No — it compiles and then faults** | Measured here: `illegal memory access was encountered` |
| Does passing by pointer work? | **Compiles, but it is the form that already broke graph capture here** | Programming Guide §4.12.2.2 lists global memory as an alternative; port commit `1218d574` |
| Does dropping `__grid_constant__` cost performance? | **Yes, structurally — measured in the PTX** | 512-byte per-thread local copy plus a 64-iteration copy loop |
| Does the fix cost performance? | **No. Zero device-side difference** | The two PTX bodies are instruction-identical; only `.param .align` differs |
| Is the `ld.global.nc` argument valid? | **No — it does not apply to these kernels** | The guarantee is documented for `__restrict__` pointers, not `__grid_constant__`; the descriptor bytes are never loaded as data |
| Does CUTLASS corroborate? | **Yes, and it is the same fix** | CUTLASS aliases the descriptor to `CUtensorMap` and never adds its own `alignas` |
| Is there a compiler flag? | **No** | `/Zp` stops at 16; C2719 is an error, so `/wd` cannot demote it; `nvcc --help` has no alignment or kernel-ABI option |

---

## 1. Is this a limitation or a bug?

**A hard limit of the MSVC x64 ABI, and nvcc is reporting it faithfully.** The decisive sentence
is in Microsoft's calling-convention documentation:

> "Most structures are aligned to their natural alignment. The primary exceptions are the stack
> pointer and `malloc` or `alloca` memory, which are 16-byte aligned to aid performance. **Alignment
> above 16 bytes must be done manually.**"
> — [x64 Calling Convention, §Alignment](https://learn.microsoft.com/en-us/cpp/build/x64-calling-convention)

The same page caps caller-allocated parameter temporaries at the same value: "For these aggregate
types passed as a pointer, including `__m128`, the caller-allocated temporary memory must be
16-byte aligned." And stack parameters are written with "8-byte alignment for each argument". The
type-layout table in [x64 software conventions](https://learn.microsoft.com/en-us/cpp/build/x64-software-conventions)
tops out at "Octaword — 128 bits" for `__m128`; there is no 128-byte entry. `/Zp` accepts only
`1 | 2 | 4 | 8 | 16` ([/Zp (Struct Member Alignment)](https://learn.microsoft.com/en-us/cpp/build/reference/zp-struct-member-alignment)).

MSVC's own error documentation says the same thing without committing to a number:

> "'parameter': formal parameter with `__declspec(align('#'))` won't be aligned. The `__declspec`
> modifier is not permitted on function parameters. **Function parameter alignment is controlled by
> the calling convention used.**"
> — [Compiler Error C2719](https://learn.microsoft.com/en-us/cpp/error-messages/compiler-errors-2/compiler-error-c2719)

So the distinction the brief asks for resolves as: *MSVC cannot represent this ABI*. It is not a
front-end slip that could be patched, because there is no 128-byte-aligned parameter slot in the
convention to emit into. A "bug" report against MSVC would be asking for a new calling convention.

**The exact threshold is 64, and that is not a coincidence.** I swept a 512-byte `CUtensorMap`
aggregate through `alignas(N)` for a by-value `__grid_constant__` parameter, compiled with this
port's toolchain (CUDA 13.3 + MSVC 14.51, `sm_120a`):

| `alignas(N)` on the aggregate | Result |
|---|---|
| 8, 16, 32, 64 | accepted |
| 128, 256 | **rejected, C2719** |

*Measured here, 2026-09-28.* The acceptance of 64 and rejection of 128 is exactly why NVIDIA's
shipped header lowers the value under MSVC (`C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3\include\cuda.h:3749-3762`):

```c
#if defined(_MSC_VER)
  #define TENSOR_MAP_ALIGN 64
#else
  #define TENSOR_MAP_ALIGN 128
#endif
typedef struct CUtensorMap_st {
#if defined(__cplusplus) && (__cplusplus >= 201103L)
    alignas(TENSOR_MAP_ALIGN)
#elif __STDC_VERSION__ >= 201112L
    _Alignas(TENSOR_MAP_ALIGN)
#endif
    cuuint64_t opaque[CU_TENSOR_MAP_NUM_QWORDS];
} CUtensorMap;
```

NVIDIA therefore already encodes the MSVC answer in the toolkit the port uses. The wrapper struct's
`alignas(128)` overrides that per-compiler decision with a hard-coded one, and the override is what
breaks. The reference page states the general requirement as "Requires compiler support for aligning
to 128 bytes" ([CUDA Driver API: `CUtensorMap`](https://docs.nvidia.com/cuda/cuda-driver-api/structCUtensorMap.html)),
i.e. a requirement on the *compiler*, which the MSVC branch is NVIDIA conceding it does not meet.

**Reproduced locally.** Compiling upstream's declaration verbatim produces the reported error, and
removing only the `alignas` clears it:

```
=== A: alignas(128) by value, __grid_constant__ ===
probe_a128.cu(12): error C2719: 'd': formal parameter with requested alignment of 128 won't be aligned
...\probe_a128.compute_120a.cudafe1.stub.c(19): error C2719: 'unnamed-parameter': formal parameter with requested alignment of 128 won't be aligned
...\probe_a128.compute_120a.cudafe1.stub.c(21): error C2719: '__cuda_0': formal parameter with requested alignment of 128 won't be aligned
exit=2
=== B: no override, by value, __grid_constant__ ===   exit=0
=== C: by const T&, no override ===                    exit=0
```

Note *where* the error lands: `...cudafe1.cpp` and `...cudafe1.stub.c` are nvcc's generated host
files, handed to MSVC. The device compilation succeeded. `alignas(128)` is a statement about the
host function's parameter, and only the host function.

## 2. The canonical fix

**Remove the `alignas`; keep by-value `__grid_constant__`.** NVIDIA states this directly, in the
section on exactly this hardware feature:

> "There are three ways to make a tensor map accessible to device code. **The recommended approach
> is to pass the tensor map as a const `__grid_constant__` parameter to a kernel.** The other
> possibilities are copying the tensor map into device `__constant__` memory using
> `cudaMemcpyToSymbol` or accessing it via global memory. When passing the tensor map as a
> parameter, some versions of the GCC C++ compiler issue the warning 'the ABI for passing parameters
> with 64-byte alignment has changed in GCC 4.6'. This warning can be ignored."
> — [CUDA Programming Guide §4.12.2.2, *Using TMA to transfer multi-dimensional arrays*](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/async-copies.html)

That last sentence is worth reading twice: NVIDIA documents the by-value parameter as carrying
**64-byte** ABI alignment. `TENSOR_MAP_ALIGN` is 64 under MSVC. The fix does not degrade the
parameter below NVIDIA's own documented expectation; it matches it.

The candidates, against that primary source:

| Option | Compiles on MSVC? | Correct? | Verdict |
|---|---|---|---|
| by value + `__grid_constant__`, no `alignas` override | yes | yes | **Recommended.** NVIDIA's documented form; CUTLASS's form; the port's existing form |
| `const T&` | yes | **no** | Rejected on measurement — see §3 |
| by pointer | yes | yes, but | NVIDIA lists it as an alternative ("global memory"), and it is the form that broke this port's graph capture in `1218d574` |
| by value, drop `__grid_constant__` | yes | yes | Works, but adds a per-thread copy — see §3 |
| `const T&`, drop `__grid_constant__` | yes | **no** | Combines two defects |
| remove `alignas`, align the pointer inside | n/a | n/a | Not applicable — there is no pointer |

**The `__grid_constant__` requirement is a language rule, not a style preference:** "Kernel
parameters annotated with `__grid_constant__` must have `const`-qualified non-reference types"
([CUDA Programming Guide §5.4.1.5](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-extensions.html)).
Passing by reference is therefore not an option that keeps the qualifier — it is an option that
discards it, which is why it must be judged on its own merits. And it fails on its own merits.

**If a 128-byte-aligned value is genuinely wanted** — for a host-side staging buffer, say — put the
alignment on the *variable*, not the type. `alignas(128) Fp8TmaDescriptors d;` gives the object the
alignment without changing `alignof(Fp8TmaDescriptors)`, so the kernel parameter stays legal. No
site in this port needs it: the descriptor is built on the stack and handed straight to the launch.

## 3. Does the fix cost performance?

**No — and this is measured, not argued.** The `alignas` has no device-side effect whatsoever. I
compiled the two forms to PTX for `sm_120a` and compared the bodies (CUDA 13.3):

```ptx
.visible .entry _Z10k_byval_gc5Desc4(              // port's form, no override
	.param .align 8 .b8 _Z10k_byval_gc5Desc4_param_0[512]
)
	mov.b64 	%rd3, _Z10k_byval_gc5Desc4_param_0;
	cvta.param.u64 	%rd1, %rd3;
	prefetch.tensormap [%rd1];
	add.s64 	%rd2, %rd1, 384;
	prefetch.tensormap [%rd2];
	ret;

.visible .entry _Z15k_byval_gc_a1289Desc4A128(     // upstream's form, alignas(128)
	.param .align 128 .b8 _Z15k_byval_gc_a1289Desc4A128_param_0[512]
)
	mov.b64 	%rd3, _Z15k_byval_gc_a1289Desc4A128_param_0;
	cvta.param.u64 	%rd1, %rd3;
	prefetch.tensormap [%rd1];
	add.s64 	%rd2, %rd1, 384;
	prefetch.tensormap [%rd2];
	ret;
```

Instruction-for-instruction identical. The only difference is `.param .align 8` versus
`.param .align 128` — the declaration MSVC then refuses to honour. *Measured here, 2026-09-28.*
There is nothing for the fix to cost, because the thing being removed was never in the device code.

### The by-reference form compiles and then faults

This is the finding that most changes the answer, and it is not visible from any documentation.
`const T&` is accepted by nvcc with the descriptor's alignment untouched, and it is the option
whose "loss of `__grid_constant__`" sounds like the mildest of the four. It is not. The generated
host stub passes the **host** address of the caller's object as a device pointer:

```c
void __device_stub__Z1kRK5Desc4i(const struct Desc4 *__par0, int __par1)
{
__cudaLaunchPrologue(2);
__cudaSetupArgSimple(__par0, 0Ui64);      // 8 bytes: the pointer value itself
__cudaSetupArgSimple(__par1, 8Ui64);
__cudaLaunch(((char *)((void ( *)(const struct Desc4 &, int))k)));
}
void k( const struct ::Desc4 &__cuda_0, int __cuda_1)
{__device_stub__Z1kRK5Desc4i( __cudaAddressOf(__cuda_0),__cuda_1);
}
```

`__cudaAddressOf` is a plain host-address cast (`CUDA 13.3 include/crt/host_runtime.h:63-67`):

```c
template <typename T>
static inline T *__cudaAddressOf(T &val)
{
  return (T *)((void *)(&(const_cast<char &>(reinterpret_cast<const volatile char &>(val)))));
}
```

Compare the by-value `__grid_constant__` stub for the same struct, which copies the 512 bytes into
the launch parameter buffer:

```c
void __device_stub__Z1k9Desc4A128i(const struct Desc4A128&__par0, int __par1)
{
__cudaLaunchPrologue(2);
__cudaSetupArg(__par0, 0Ui64);            // 512 bytes staged at offset 0
__cudaSetupArgSimple(__par1, 512Ui64);
__cudaLaunch(((char *)((void ( *)(const struct Desc4A128, int))k)));
}
```

*Read in the source: generated by `nvcc -keep` on this machine, CUDA 13.3.* Running both on the
5090 makes the consequence plain:

```
byval+grid_constant: launch=no error sync=no error read=no error
  -> value=deadbeef (expect deadbeef) same-object=1
by const T&:         launch=no error
  sync=an illegal memory access was encountered
```

*Measured here, 2026-09-28.* Note `launch=no error` for the by-reference form: the launch is
accepted and the failure appears at synchronize, so this is the shape of bug that survives a
compile-only check. `same-object=1` in the working case is the `__grid_constant__` guarantee
holding — all 32 threads observed one object.

### What dropping `__grid_constant__` actually costs

By value *without* the qualifier, the compiler must give each thread its own copy, because C++
allows a thread to take the address of its parameter and modify it. The PTX shows the cost exactly:

```ptx
.visible .entry _Z12k_byval_nogc5Desc4(
	.param .align 8 .b8 _Z12k_byval_nogc5Desc4_param_0[512]
)
	.local .align 8 .b8 	__local_depot3[512];
	mov.b64 	%SPL, __local_depot3;
	mov.b64 	%rd4, _Z12k_byval_nogc5Desc4_param_0;
$L__BB3_1:
	ld.param.b64 	%rd2, [%rd4];
	st.local.b64 	[%rd5], %rd2;
	...
	@%p1 bra 	$L__BB3_1;                // 64 iterations: 512 bytes, per thread
```

*Measured here, 2026-09-28.* 512 bytes of local frame and a 64-iteration copy loop per thread,
against zero in the `__grid_constant__` form. This is a structural cost, and it is not a rounding
error: at 256 threads/block the local frame is 128 KiB per block. NVIDIA describes the same
mechanism — "if you omit the `__grid_constant__` qualifier to the kernel parameter and perform a
subsequent write operation to it from the kernel, an automatic copy to `thread-local-memory` is
triggered. This may offset any performance gains"
([CUDA 12.1 Supports Large Kernel Parameters](https://developer.nvidia.com/blog/cuda-12-1-supports-large-kernel-parameters)).

This matters for these kernels in particular because the descriptor's address *is* taken, on every
load: `nvfp4_tma_load_2d(tensors.a_codes[stage], &descriptors.a_codes, ...)`
(`src/ops/linear/nvfp4/nvfp4_a4_tma.cuh:202`), and likewise at lines 209, 218, 227 and in
`src/ops/linear/bf16/bf16_a16_tma_mma.cuh:109,111`. That is precisely the case the qualifier exists
for.

### The `ld.global.nc` premise in the brief does not hold

The brief states that `__grid_constant__` "enabl[es] non-coherent (`ld.global.nc` / `__ldg`)
loads". The Programming Guide makes that promise for a *different* annotation:

> "Accesses to `__global__` function `const` pointers marked with `__restrict__` are compiled as
> read-only cache loads, similar to the PTX `ld.global.nc` or `__ldg()` low-level load and store
> functions instructions."
> — [CUDA Programming Guide §5.4.1.4, *`__restrict__` Pointers*](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-extensions.html)

`__grid_constant__` gets a different and more specific promise, in the adjacent section:

> "Annotating a `__global__` function parameter with `__grid_constant__` prevents the compiler from
> creating a per-thread copy of the parameter. Instead, all threads in the grid will access the
> parameter through a single address, which can improve performance."
> — [CUDA Programming Guide §5.4.1.5, *`__grid_constant__` Parameters*](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-extensions.html)

And for a TMA descriptor the question is moot in any case: the bytes are never loaded as data. They
are handed to the TMA unit as an address by `cp.async.bulk.tensor.*`, as the PTX above shows
(`prefetch.tensormap [%rd1]` — no load instruction at all). So there is no `ld.global.nc` to lose
under any of the four options. The documented benefit at stake is the absence of the per-thread
copy, and that is the one the PTX measures.

**No published figure exists for a by-reference performance cost.** I found none in the Programming
Guide, the nvcc documentation, the CUDA 12.1 blog, or the CUTLASS tree. It does not need one: the
by-reference form is broken, so its speed is not a trade-off. What is measured above is the cost of
the one *correct* alternative (dropping `__grid_constant__`), and it is the reason not to take it.

## 4. CUTLASS corroborates, with the same rule

CUTLASS 3.x uses TMA descriptors heavily, claims MSVC support ("CUTLASS 3.2 reintroduces support
for the Microsoft Visual Studio compiler on Windows",
[Building on Windows with Visual Studio](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/build/building_in_windows_with_visual_studio.html)),
and passes its descriptor struct to `__global__` kernels **by value** under `__grid_constant__` —
the same shape as this port and the same shape as upstream. It does not hit C2719 because it never
adds its own `alignas`. Two declarations, quoted from `main`:

`include/cute/arch/copy_sm90_desc.hpp`:

```c++
#if (__CUDACC_VER_MAJOR__ >= 12) && !defined(__CUDACC_RTC__)
  using TmaDescriptor = CUtensorMap;
  using Im2ColTmaDescriptor = CUtensorMap;
#else
  using TmaDescriptor = struct alignas(64) { char bytes[128]; };
  using Im2ColTmaDescriptor = struct alignas(64) { char bytes[128]; };
#endif
```

Two things to notice. CUTLASS aliases the descriptor to `CUtensorMap` outright, inheriting
`TENSOR_MAP_ALIGN` and therefore the per-compiler 64/128 decision. And its own fallback type for a
128-byte descriptor payload is `alignas(64)`, not `alignas(128)` — NVIDIA's own libraries do not ask
for 128 from a user-declared type either.

The descriptor then travels as an ordinary member, with no added alignment
(`include/cute/atom/copy_traits_sm90_tma.hpp`):

```c++
template <class NumBitsPerTMA, class AuxParams_>
struct Copy_Traits<SM90_TMA_LOAD, NumBitsPerTMA, AuxParams_>
{
  using ThrID     = Layout<_1>;
  // ...
  // SM90_TMA_LOAD arguments
  TmaDescriptor tma_desc_;
  using AuxParams = AuxParams_;
  AuxParams aux_params_;
```

and the aggregate reaches the kernel by value (`include/cutlass/device_kernel.h`):

```c++
// __grid_constant__ was introduced in CUDA 11.7.
#if ((__CUDACC_VER_MAJOR__ >= 12) || ((__CUDACC_VER_MAJOR__ == 11) && (__CUDACC_VER_MINOR__ >= 7))) && !CUTLASS_CLANG_CUDA
#  define CUTLASS_GRID_CONSTANT_SUPPORTED
#endif

// __grid_constant__ can be enabled only on SM70+
#if defined(CUTLASS_GRID_CONSTANT_SUPPORTED) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 700)
#  define CUTLASS_GRID_CONSTANT_ENABLED
#endif

#if ! defined(CUTLASS_GRID_CONSTANT)
#  if defined(CUTLASS_GRID_CONSTANT_ENABLED)
#    define CUTLASS_GRID_CONSTANT __grid_constant__
#  else
#    define CUTLASS_GRID_CONSTANT
#  endif
#endif
```

```c++
CUTLASS_GLOBAL
#ifdef __CUDACC__
// Enclosing this in __CUDACC__ suppresses MSVC warnings.
__launch_bounds__(Operator::MaxThreadsPerBlock, Operator::MinBlocksPerMultiprocessor)
#endif // __CUDACC__
void device_kernel(CUTLASS_GRID_CONSTANT typename Operator::Params const params)
{
  extern __shared__ char smem[];
  Operator op;
  op(params, smem);
  cutlass::arch::synclog_print();
}
```

The same by-value form appears in the CuTe tutorial kernel
(`examples/cute/tutorial/hopper/wgmma_tma_sm90.cu`): `TA const* A, CUTLASS_GRID_CONSTANT TmaA const
tma_a,`. The CuTe DSL carries the same recommendation in prose — "Use `GridConstant[TensorMap]` for
kernel parameters that carry TMA descriptors. This marks the argument as `__grid_constant__` so that
the descriptor lives in constant memory"
([cute.CUDA](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cuda.html)).

So CUTLASS's answer to this exact problem is: inherit the compiler's alignment, pass by value under
`__grid_constant__`, and never name 128 in user code. That is the recommendation at the top of this
note, arrived at independently.

**What I could not establish.** I did not locate a declaration of the SM90 collective-builder's
`cute::TMA` aggregate in CUTLASS `main` to confirm its own `alignas`. GitHub's code-search API
returned empty for queries that should have matched (`TmaDescriptor repo:NVIDIA/cutlass` → 0), so
the search index could not be used to rule it in or out, and the type is not in
`include/cute/atom/copy_traits_sm90_tma.hpp` or `include/cutlass/gemm/collective/builders/sm90_common.inl`.
This does not affect the conclusion: `device_kernel` passes `Operator::Params` by value under
`CUTLASS_GRID_CONSTANT`, and if any member of that struct were 128-aligned on MSVC, CUTLASS would
fail to build there, which it does not. But the claim is a corollary, not a direct reading, and is
labelled as such.

## 5. Other reports

**One independent third-party report, and it corroborates the diagnosis.** Microsoft's ONNX Runtime
hit the identical error on its Windows CUDA 13.2/13.3 packaging pipeline, in
[commit `291311e7`](https://github.com/microsoft/onnxruntime/commit/291311e7d8d4ac17672e6bdaffaa4a8539115e87)
("Fix CUDA 13 nuget packaging pipeline failures and enable CUDA 13.3 builds (#28736)"):

> "**CUDA 13.2/13.3 compiler incompatibilities**: NVCC 13.x generates host stubs that pass TMA
> descriptor parameters (with `alignas(128)`) by value. MSVC rejects this with error C2719."

*Reported by a third party; quoted from the commit message.* Their fix was heavier than this one —
they wrapped `COMPILE_HOPPER_TMA_GROUPED_GEMMS` and `COMPILE_BLACKWELL_SM120_TMA_GROUPED_GEMMS` in
`if(NOT MSVC)` in `cmake/onnxruntime_providers_cuda.cmake` and added a runtime feature check so
standard MoE falls back to SM80 Ampere kernels on MSVC, with a clear error for the TMA-only modes.
That is a legitimate choice for a project that does not want to patch vendored CUTLASS-derived
kernels; it is not a statement that the error is unfixable in place. Their diagnosis — host stub,
`alignas(128)`, by value, C2719 — is the same as the one reached here from the PTX and the
generated stub.

**No NVIDIA bug-tracker entry found.** I searched the NVIDIA developer forums and the NVIDIA GitHub
repositories for `C2719` with `alignas`, `__grid_constant__`, or tensor-map terms and did not find a
report filed against nvcc. I would not claim none exists; I can say I did not find one. The
practical consequence is that there is no upstream fix to wait for, and the header's `_MSC_VER`
branch is the standing answer.

**Nothing found in nvcc release notes** for CUDA 13.2 or 13.3 that changes this. The `_MSC_VER`
branch in `cuda.h` is the mechanism NVIDIA ships.

## 6. Compiler flags

**No flag avoids the problem, and one tempting one is a trap.**

- **`/Zp`** takes `1 | 2 | 4 | 8 | 16` and nothing above ([docs](https://learn.microsoft.com/en-us/cpp/build/reference/zp-struct-member-alignment)). It also packs *members*, not parameters, and the docs warn that changing it breaks the Windows SDK's assumptions. It cannot express 128, and it would not help at 64 either.
- **`/wd2719` cannot work.** The reproduced diagnostic is `error C2719`, not a warning; there is nothing to demote. *(Measured here — the `error` prefix is in the compiler output quoted in §1.)*
- **`/d2…` alignment options.** None documented for over-aligned parameters. The x64 convention's 16-byte ceiling is a property of the ABI, not a front-end policy switch.
- **`nvcc --help`** has no alignment, parameter, or kernel-ABI option. The `--expt-*` family is extended-lambda and language-feature control; it does not change how the host stub declares a parameter. `--no-align-double` is the only alignment-named option and is documented as a 32-bit host-compiler flag that "makes the ABI incompatible with the cuda's kernel ABI for certain 64-bit types" — the wrong direction entirely.
- **`-Xcompiler`** forwards flags to the host compiler, so it can only reach the switches above, none of which apply.

The header branch is the sanctioned mechanism, and it is already in the toolkit the port builds
against.

## What I did not check

- Whether MSVC would accept 128 under a different calling convention (`__vectorcall`, `__preserve_none`). The x64 convention documentation is scoped to the default convention, and CUDA's host stub uses the default, so this is academic — but I did not test it.
- Performance of the TMA kernels with and without `__grid_constant__` at the engine level. The PTX establishes the structural difference (a 512-byte per-thread local copy and a 64-iteration loop); it does not quantify the end-to-end effect on a decode round, which would need an interleaved A/B on this card. Not required for the recommendation, since the recommended form is the one already in the tree.
- The Blackwell driver bug that requires clearing bit 85 of the descriptor for tensors under 128 KB, documented in the Tesla release notes and worked around in CUTLASS (`reinterpret_cast<uint64_t*>(&tma_desc)[1] &= ~(1llu << 21);` in `include/cute/atom/copy_traits_sm90_tma.hpp`) and in [triton #10442](https://github.com/triton-lang/triton/issues/10442). Unrelated to C2719, but it is a second descriptor-layout trap on this hardware and this port does not carry the workaround. Flagged, not investigated.

## Sources

Primary:

- Microsoft, [Compiler Error C2719](https://learn.microsoft.com/en-us/cpp/error-messages/compiler-errors-2/compiler-error-c2719)
- Microsoft, [x64 Calling Convention](https://learn.microsoft.com/en-us/cpp/build/x64-calling-convention) — §Alignment, §Parameter passing
- Microsoft, [x64 software conventions](https://learn.microsoft.com/en-us/cpp/build/x64-software-conventions) — §x64 type and storage layout
- Microsoft, [/Zp (Struct Member Alignment)](https://learn.microsoft.com/en-us/cpp/build/reference/zp-struct-member-alignment)
- NVIDIA, [CUDA Programming Guide §4.12.2.2, *Using TMA to transfer multi-dimensional arrays*](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/async-copies.html) — "Host-to-device transfer"
- NVIDIA, [CUDA Programming Guide §5.4.1.4, `__restrict__` Pointers](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-extensions.html)
- NVIDIA, [CUDA Programming Guide §5.4.1.5, `__grid_constant__` Parameters](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-extensions.html)
- NVIDIA, [CUDA 12.1 Supports Large Kernel Parameters](https://developer.nvidia.com/blog/cuda-12-1-supports-large-kernel-parameters)
- NVIDIA, [CUDA Driver API: `CUtensorMap`](https://docs.nvidia.com/cuda/cuda-driver-api/structCUtensorMap.html)
- NVIDIA, [Building on Windows with Visual Studio (CUTLASS)](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/build/building_in_windows_with_visual_studio.html)
- NVIDIA, [cute.CUDA (CuTe DSL)](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cuda.html) — `GridConstant[TensorMap]`
- CUTLASS `main`, `include/cute/arch/copy_sm90_desc.hpp`, `include/cute/atom/copy_traits_sm90_tma.hpp`, `include/cutlass/device_kernel.h`, `examples/cute/tutorial/hopper/wgmma_tma_sm90.cu` (read via `raw.githubusercontent.com`; cited inline)
- ONNX Runtime, [commit `291311e7`](https://github.com/microsoft/onnxruntime/commit/291311e7d8d4ac17672e6bdaffaa4a8539115e87) — third-party report, quoted inline

Read locally, not served at a stable URL:

- CUDA 13.3 `include/cuda.h:3740-3762` — `TENSOR_MAP_ALIGN` and `CUtensorMap`
- CUDA 13.3 `include/crt/host_runtime.h:63-67, 101-105` — `__cudaAddressOf`, `__cudaSetupArg`, `__cudaSetupArgSimple`
- `nvcc --help` (CUDA 13.3) — full option list
- Generated `.ptx` and `.cudafe1.stub.c` from scratch probes compiled with this port's toolchain; sources and probes were temporary and are not part of the tree

In this repository:

- `upstream/dev` (`7f6aafed`) — `src/ops/linear/fp8/fp8_a8_tma_mma.cuh:20, 101-105`, `src/ops/linear/nvfp4/nvfp4_a4_tma.cuh:27`, `src/ops/linear/bf16/bf16_a16_tma_mma.cuh:20`
- Port commit `1218d574` — the same fix, previously applied and since merged
- [`docs/research/windows-port-best-practice.md`](./windows-port-best-practice.md) §1.4, §3 — the earlier note reaching this conclusion by a different route; this note supersedes its §3 evidence and corrects its attribution of the `ld.global.nc` claim
