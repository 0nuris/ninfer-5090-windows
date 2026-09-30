#!/usr/bin/env bash
# Starts ninfer-serve with the settings in ninfer.conf and waits until /health answers.
# Exit code 0 = READY, 1 = refused or failed (the reason is printed; details in logs/ninfer.err).
set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=ninfer.conf
. "$ROOT/ninfer.conf"
rooted() { case "$1" in /*) printf '%s' "$1" ;; *) printf '%s' "$ROOT/$1" ;; esac; }
EXE_PATH="$(rooted "$EXE")"
MODEL_PATH="$(rooted "$MODEL")"
LOG_DIR="$(rooted "$LOGS")"

for f in "$EXE_PATH" "$MODEL_PATH"; do
    [ -e "$f" ] || { echo "ABORT: missing $f (run install.sh)"; exit 1; }
done
mkdir -p "$LOG_DIR"

# A VPN or LAN address may not exist yet right after boot, and binding a missing address fails.
case "$BIND_ADDRESS" in
    127.0.0.1|0.0.0.0|localhost) ;;
    *)
        for _ in $(seq 60); do
            ip -o addr show | grep -qw "inet $BIND_ADDRESS" && break
            sleep 2
        done
        if ! ip -o addr show | grep -qw "inet $BIND_ADDRESS"; then
            echo "ABORT: $BIND_ADDRESS is not assigned on this machine (network or VPN down?)"
            exit 1
        fi
        ;;
esac

# Replace a previous ninfer-serve on this port, but never stop an unrelated program.
if [ -n "$(ss -ltnH "sport = :$PORT" 2>/dev/null)" ]; then
    # ss -p prints users:(("<name>",pid=<pid>,fd=<n>)) for processes this user may inspect.
    read -r name pid < <(ss -ltnpH "sport = :$PORT" 2>/dev/null | awk '
        match($0, /users:\(\("[^"]*",pid=[0-9]+/) {
            s = substr($0, RSTART + 9, RLENGTH - 9)       # <name>",pid=<pid>
            split(s, part, "\",pid=")
            print part[1], part[2]
            exit
        }')
    if [ "${name:-}" != "ninfer-serve" ] || [ -z "${pid:-}" ]; then
        holder="a process owned by another user"
        [ -n "${name:-}" ] && holder="$name (pid $pid)"
        echo "ABORT: port $PORT is in use by $holder. Stop it or set PORT in ninfer.conf."
        exit 1
    fi
    kill "$pid" 2>/dev/null
    for _ in $(seq 20); do kill -0 "$pid" 2>/dev/null || break; sleep 0.5; done
    kill -9 "$pid" 2>/dev/null
    sleep 2   # let the driver release its VRAM before the check below
fi

# Advisory only: the engine sizes its KV pool from the VRAM actually free and refuses with an
# exact "minimum Engine runtime reservation" message if the profile does not fit.
free_mib="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -n1 | tr -dc '0-9')"
if [ -n "$free_mib" ]; then
    free_gib="$(awk -v m="$free_mib" 'BEGIN { printf "%.1f", m / 1024 }')"
    echo "VRAM free ${free_gib} GiB (this profile typically uses ~${NEED_VRAM_GIB} GiB)"
    if awk -v f="$free_gib" -v n="$NEED_VRAM_GIB" 'BEGIN { exit !(f < n) }'; then
        echo "WARN: less VRAM free than this profile typically uses. If startup fails, close other GPU programs or lower MAX_CONTEXT / CONCURRENCY / HOST_KV_MIB."
    fi
fi

# Keep the previous run's log instead of overwriting it.
for name in ninfer.err ninfer.out; do
    current="$LOG_DIR/$name"
    if [ -s "$current" ]; then
        stamp="$(date -r "$current" +%Y%m%d-%H%M%S)"
        mv -f "$current" "$LOG_DIR/${name%.*}-$stamp.${name##*.}"
    fi
    ls -1t "$LOG_DIR/${name%.*}"-*."${name##*.}" 2>/dev/null | tail -n +"$((KEEP_LOGS + 1))" | xargs -r rm -f
done

# Bundled libraries (the CUDA runtime) sit next to the engine.
export LD_LIBRARY_PATH="$(dirname "$EXE_PATH")${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

# setsid + redirected stdio: the server inherits no terminal or pipe from the caller, so a
# caller reading this script's output (tee, a service manager) sees it finish.
setsid "$EXE_PATH" "$MODEL_PATH" \
    --spec "$SPEC" --draft-tokens "$DRAFT_TOKENS" --lm-head-draft \
    --host "$BIND_ADDRESS" --port "$PORT" --model-id "$MODEL_ID" \
    --max-context "$MAX_CONTEXT" --kv-capacity auto --kv-dtype "$KV_DTYPE" \
    --device-state-slots 1 --prefill-chunk 8192 \
    --max-concurrency "$CONCURRENCY" \
    --host-state-slots "$HOST_STATE_SLOTS" --host-kv-mib "$HOST_KV_MIB" \
    --max-shared-prefixes 7 --max-private-continuations 8 \
    --max-long-anchors-per-continuation 4 \
    --preserve-thinking --default-thinking-budget "$THINKING_BUDGET" \
    --pending-timeout-ms 600000 \
    </dev/null >"$LOG_DIR/ninfer.out" 2>"$LOG_DIR/ninfer.err" &
server_pid=$!
echo "ninfer starting (pid $server_pid)"

probe="$BIND_ADDRESS"
[ "$probe" = "0.0.0.0" ] && probe="127.0.0.1"
for _ in $(seq 90); do
    sleep 2
    if ! kill -0 "$server_pid" 2>/dev/null; then
        echo "process exited early - see $LOG_DIR/ninfer.err"
        echo "ninfer on $BIND_ADDRESS:$PORT -> FAILED"
        exit 1
    fi
    if curl -fsS -m 3 "http://$probe:$PORT/health" >/dev/null 2>&1; then
        echo "ninfer on $BIND_ADDRESS:$PORT -> READY"
        exit 0
    fi
done
echo "ninfer on $BIND_ADDRESS:$PORT -> FAILED (no /health after 180 s; see $LOG_DIR/ninfer.err)"
exit 1
