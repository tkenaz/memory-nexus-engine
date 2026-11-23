#!/usr/bin/env bash
# Run the whole suite (unit + integration) against a throwaway pgvector container.
#
# Starts pgvector/pgvector:pg16 under a random name (memory-nexus-test-<hex>) on a
# free high port chosen by Docker, waits until it accepts TCP connections, runs
# pytest, and removes exactly that container on exit (also on failure or Ctrl-C).
# No other container is touched.
#
#   tests/run_with_docker.sh                      # everything
#   tests/run_with_docker.sh tests/integration -x # extra args go to pytest
#   PYTHON=/path/to/venv/bin/python tests/run_with_docker.sh
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PYTHON:-python3}"
IMAGE="${PGVECTOR_IMAGE:-pgvector/pgvector:pg16}"
NAME="memory-nexus-test-$(od -An -N4 -tx1 /dev/urandom | tr -d ' \n')"

cleanup() { docker rm -f "$NAME" >/dev/null 2>&1 || true; }
trap cleanup EXIT

docker run -d --name "$NAME" \
    -e POSTGRES_USER=postgres -e POSTGRES_PASSWORD=test -e POSTGRES_DB=memory_nexus_test \
    -p 127.0.0.1::5432 "$IMAGE" >/dev/null
PORT="$(docker port "$NAME" 5432/tcp | sed -n '1s/.*://p')"  # sed reads all input: no SIGPIPE under pipefail

# The image's init phase listens on a unix socket only; TCP readiness = real start.
for _ in $(seq 1 120); do
    docker exec "$NAME" pg_isready -h 127.0.0.1 -U postgres -d memory_nexus_test >/dev/null 2>&1 && break
    sleep 0.5
done

export MEMORY_NEXUS_TEST_DSN="postgresql://postgres:test@127.0.0.1:${PORT}/memory_nexus_test"
echo "throwaway container ${NAME} on 127.0.0.1:${PORT}"

# Default to the whole suite unless a test path is among the arguments
# (flags alone, such as -q -rx, must not make pytest collect from the cwd).
has_path=0
for arg in "$@"; do
    case "$arg" in -*) ;; *) [ -e "${arg%%::*}" ] && has_path=1 ;; esac
done
if [ "$has_path" -eq 0 ]; then
    set -- "$@" "$HERE"
fi
"$PY" -m pytest -c "$HERE/pytest.ini" "$@"
