#!/usr/bin/env bash
# Stops everything scripts/demo.sh started (kills each process tree via its pid file).
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
RUN="$ROOT/.demo"

kill_tree() {
  for child in $(pgrep -P "$1" 2>/dev/null); do kill_tree "$child"; done
  kill "$1" 2>/dev/null
}

for name in console gateway api; do
  f="$RUN/$name.pid"
  if [ -f "$f" ]; then
    kill_tree "$(cat "$f")"
    rm -f "$f"
    echo "stopped $name"
  fi
done
echo "done"
