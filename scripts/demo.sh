#!/usr/bin/env bash
# One-command demo: agent API (Groq drafter), GraphQL gateway, console.
# macOS bash 3.2 / zsh compatible. Stop with scripts/demo_stop.sh.
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
RUN="$ROOT/.demo"
mkdir -p "$RUN"
cd "$ROOT" || exit 1

die() { echo "ERROR: $*" >&2; exit 1; }

[ -x "$ROOT/.venv/bin/uvicorn" ] || die ".venv missing or incomplete. Run: python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt langchain-groq"
grep -q '^GROQ_API_KEY=.' "$ROOT/.env" 2>/dev/null || echo "WARN: no GROQ_API_KEY in .env, the demo will run on the extractive fallback."

for port in 8000 4000 5173; do
  if lsof -nP -iTCP:"$port" -sTCP:LISTEN >/dev/null 2>&1; then
    die "port $port is already in use. Press Ctrl+C in your old Terminal tabs (or run scripts/demo_stop.sh), then retry."
  fi
done

[ -d gateway/node_modules ] || (cd gateway && npm ci --silent) || die "npm ci failed in gateway"
[ -d console/node_modules ] || (cd console && npm ci --silent) || die "npm ci failed in console"

start() { # name, workdir, command...
  name="$1"; dir="$2"; shift 2
  (cd "$dir" && nohup "$@" </dev/null >"$RUN/$name.log" 2>&1 & echo $! >"$RUN/$name.pid")
}

wait_for() { # label, url, seconds, logname
  i=0
  while [ "$i" -lt "$3" ]; do
    curl -fsS -o /dev/null "$2" 2>/dev/null && { echo "  ok  $1"; return 0; }
    sleep 1; i=$((i + 1))
  done
  echo "  FAILED  $1 (see $RUN/$4.log)" >&2
  return 1
}

echo "Starting agent API (ARC_DRAFTER=groq ARC_SMALL=1)..."
ARC_DRAFTER=groq ARC_SMALL=1 start api "$ROOT" "$ROOT/.venv/bin/uvicorn" arc.api:app --port 8000
echo "Starting gateway..."
start gateway "$ROOT/gateway" npm start
echo "Starting console..."
start console "$ROOT/console" npm run dev -- --port 5173 --strictPort

ok=0
wait_for "agent API  :8000" "http://127.0.0.1:8000/docs" 120 api || ok=1
wait_for "gateway    :4000" "http://127.0.0.1:4000/graphql?query=%7B__typename%7D" 60 gateway || ok=1
wait_for "console    :5173" "http://localhost:5173/" 60 console || ok=1
if [ "$ok" -ne 0 ]; then
  "$ROOT/scripts/demo_stop.sh" >/dev/null 2>&1
  die "a service did not become healthy; everything was stopped. Logs are in .demo/"
fi

echo
echo "Demo ready: http://localhost:5173"
echo "Stop it with: scripts/demo_stop.sh   (logs: .demo/*.log)"
