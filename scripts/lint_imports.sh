#!/usr/bin/env bash
# The package depends on langchain-core only: no imports from langchain,
# langchain-community or langchain-experimental.
set -euo pipefail

if git --no-pager grep -nE \
  '^(from|import) langchain(_community|_experimental)?([. ]|$)' \
  -- langchain_string; then
  exit 1
fi
