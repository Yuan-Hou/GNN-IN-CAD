#!/usr/bin/env bash
set -euo pipefail

DATA_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEST="${DATA_DIR}/MFCADDataset"
URL="https://uv-net-data.s3.us-west-2.amazonaws.com/MFCADDataset.zip"
WORKDIR="$(mktemp -d)"
ZIP="${WORKDIR}/MFCADDataset.zip"

cleanup() { rm -rf "${WORKDIR}"; }
trap cleanup EXIT

if ! command -v 7z >/dev/null 2>&1 && ! command -v 7zz >/dev/null 2>&1; then
  echo "7z is required to unpack graph.7z and labels.7z. Install it with: brew install p7zip" >&2
  exit 1
fi

echo "Downloading ${URL}"
curl -L --fail --output "${ZIP}" "${URL}"

echo "Unpacking zip into ${DEST}"
unzip -q -o "${ZIP}" -d "${WORKDIR}/unzipped"

# The published zip has no wrapper directory. graph.7z, labels.7z, and
# split.json sit at the archive root. Keep a fallback if a mirror adds one.
if [[ -d "${WORKDIR}/unzipped/MFCADDataset" ]]; then
  SRC="${WORKDIR}/unzipped/MFCADDataset"
else
  SRC="${WORKDIR}/unzipped"
fi

mkdir -p "${DEST}"
cp -R "${SRC}/." "${DEST}/"

SEVENZ="$(command -v 7zz || command -v 7z)"
echo "Unpacking graph.7z and labels.7z into ${DEST}"
"${SEVENZ}" x -y "-o${DEST}" "${DEST}/graph.7z"
"${SEVENZ}" x -y "-o${DEST}" "${DEST}/labels.7z"

echo "MFCAD ready at ${DEST}"
