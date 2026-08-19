#!/usr/bin/env bash
set -uo pipefail

declare -a shell_files=()

if [ "$#" -gt 0 ]; then
  shell_files=("$@")
else
  git rev-parse --is-inside-work-tree >/dev/null 2>&1 || {
    echo "ABORT: shell syntax gate requires a Git worktree to enumerate tracked files." >&2
    exit 1
  }
  mapfile -d '' tracked_files < <(git ls-files -z)
  for path in "${tracked_files[@]}"; do
    case "$path" in
      *.sh) shell_files+=("$path") ;;
      *)
        IFS= read -r first_line < "$path" || first_line=
        case "$first_line" in
          '#!'*bash*|'#!'*/sh|'#!'*/sh' '*) shell_files+=("$path") ;;
        esac
        ;;
    esac
  done
fi

[ "${#shell_files[@]}" -gt 0 ] || {
  echo "ABORT: shell syntax gate found no files to check." >&2
  exit 1
}

failures=0
for path in "${shell_files[@]}"; do
  [ -f "$path" ] || {
    echo "SHELL SYNTAX MISSING: $path" >&2
    failures=$((failures + 1))
    continue
  }
  if ! finding=$(bash -n "$path" 2>&1); then
    echo "SHELL SYNTAX FAIL: $path" >&2
    printf '%s\n' "$finding" >&2
    failures=$((failures + 1))
  fi
done

if [ "$failures" -ne 0 ]; then
  echo "shell syntax: FAIL files=${#shell_files[@]} findings=$failures" >&2
  exit 1
fi

echo "shell syntax: PASS files=${#shell_files[@]} individually parsed"
