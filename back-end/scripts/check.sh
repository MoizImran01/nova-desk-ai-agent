#!/usr/bin/env bash
# Canonical backend checks. Run by the no-mistakes gate and by GitHub Actions.
#
#   scripts/check.sh lint   dependency + syntax checks, ruff lint, non-writing
#                           format check, declared mypy scope
#   scripts/check.sh test   pytest (unit, integration, deterministic agent regression)
#   scripts/check.sh all    both (default)
#
# `test` needs PostgreSQL and a dedicated Redis (see tests/conftest.py).
# Extend the lists below as later slices land; do not add skipped or empty checks.
set -euo pipefail

cd "$(dirname "$0")/.."
if [ -x venv/bin/python ]; then
  PATH="$PWD/venv/bin:$PATH"
fi

# Paths covered by the non-writing format check. Legacy app code is not reformatted yet.
FORMAT_PATHS=(tests)

lint() {
  python -m pip check
  python -m compileall -q app tests
  ruff check .
  ruff format --check "${FORMAT_PATHS[@]}"
  mypy
}

tests() {
  python -m pytest
}

case "${1:-all}" in
  lint) lint ;;
  test) tests ;;
  all) lint; tests ;;
  *) echo "usage: $0 [lint|test|all]" >&2; exit 2 ;;
esac

