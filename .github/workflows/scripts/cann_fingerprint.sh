#!/usr/bin/env bash
# Print a short fingerprint (12 hex chars) of the CANN toolchain installed in
# this container. The CI image registry tags are mutable (a re-synced image
# keeps its tag), so csrc build cache keys embed this fingerprint to track the
# image content actually in use instead of trusting the tag.
set -u

fp_files=()
for f in /usr/local/Ascend/ascend-toolkit/latest/version.cfg \
         /usr/local/Ascend/ascend-toolkit/latest/*/ascend_toolkit_install.info \
         /usr/local/Ascend/nnal/atb/latest/version.info; do
  if [ -f "$f" ]; then
    fp_files+=("$f")
  fi
done

if [ "${#fp_files[@]}" -gt 0 ]; then
  echo "CANN fingerprint inputs: ${fp_files[*]}" >&2
  cat "${fp_files[@]}" | sha256sum | cut -c1-12
else
  echo "::warning::no CANN version files found, falling back to toolkit path fingerprint" >&2
  { readlink -f /usr/local/Ascend/ascend-toolkit/latest; ls /usr/local/Ascend; true; } 2>/dev/null | sha256sum | cut -c1-12
fi
