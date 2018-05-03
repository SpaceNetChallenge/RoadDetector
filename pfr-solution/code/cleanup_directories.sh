set -eu

TEMP_PATH=/wdata
rm -rf "$TEMP_PATH"/computed 2>/dev/null || true
rm -f computed 2>/dev/null || true
rm -rf unpacked 2>/dev/null || true
