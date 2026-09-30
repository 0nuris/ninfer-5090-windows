#!/usr/bin/env bash
# Optional watchdog for running the server unattended. Does nothing while the server answers
# /health; otherwise runs start-ninfer.sh. Run it periodically from whatever you use to keep
# things running (a systemd timer, cron, ...); how and as whom is up to you. Each action is
# noted in logs/watchdog.log. Exit code 0 = healthy or started, 1 = start failed.
#
# To use the GPU for something else, create a file named ninfer.disabled next to this script
# before stopping the server; delete it to let the watchdog start NInfer again.
set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=ninfer.conf
. "$ROOT/ninfer.conf"
case "$LOGS" in /*) LOG_DIR="$LOGS" ;; *) LOG_DIR="$ROOT/$LOGS" ;; esac
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/watchdog.log"
probe="$BIND_ADDRESS"
[ "$probe" = "0.0.0.0" ] && probe="127.0.0.1"
URL="http://$probe:$PORT/health"

note() {
    printf '%s  %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$1" >>"$LOG"
    if [ "$(wc -l <"$LOG")" -gt 600 ]; then
        tail -n 500 "$LOG" >"$LOG.tmp" && mv -f "$LOG.tmp" "$LOG"
    fi
}

[ -e "$ROOT/ninfer.disabled" ] && exit 0
curl -fsS -m 5 "$URL" >/dev/null 2>&1 && exit 0

# /health fails for ~15 s while a server loads; give an in-progress start time to finish.
if [ -n "$(ss -ltnH "sport = :$PORT" 2>/dev/null)" ]; then
    sleep 60
    curl -fsS -m 5 "$URL" >/dev/null 2>&1 && exit 0
fi

note "health check failed; running start-ninfer.sh"
out="$("$ROOT/start-ninfer.sh" 2>&1)"
code=$?
note "start-ninfer.sh exit $code: $(printf '%s\n' "$out" | tail -n 2 | tr '\n' '|')"
exit "$code"
