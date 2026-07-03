#!/usr/bin/env bash
set -euo pipefail

readonly MAX_BYTES=94371840 # 90 MiB

echo "[hygiene] checking tracked file sizes (max 90 MiB)..."
large_files=""
while IFS= read -r tracked_file; do
  [ -f "$tracked_file" ] || continue
  size_bytes=$(stat -c%s "$tracked_file")
  if [ "$size_bytes" -gt "$MAX_BYTES" ]; then
    large_files+="$tracked_file ($size_bytes bytes)"$'\n'
  fi
done < <(git ls-files)

if [ -n "$large_files" ]; then
  echo "[hygiene] ERROR: tracked files exceed 90 MiB:" >&2
  echo "$large_files" >&2
  exit 1
fi

echo "[hygiene] checking disallowed tracked paths..."
disallowed_regex='^(data/raw/|data/models/|notebooks/data/raw/|notebooks/data/models/|problems/[^/]+/data/raw/|problems/[^/]+/data/models/)'
tracked_disallowed=$(git ls-files | grep -E "$disallowed_regex" || true)

if [ -n "$tracked_disallowed" ]; then
  echo "[hygiene] ERROR: disallowed data/model paths are tracked:" >&2
  echo "$tracked_disallowed" >&2
  exit 1
fi

echo "[hygiene] repository checks passed."
