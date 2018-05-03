set -eu

TEMP_PATH=/wdata
rm -f spacenet/* 2>/dev/null || true
rmdir spacenet 2>/dev/null || true
mkdir spacenet
for path in $@; do
    echo "Input directory: '$path'"
    ln -s "$path" spacenet/
done
rm -rf "$TEMP_PATH"/computed 2>/dev/null || true
rm -f computed 2>/dev/null || true
mkdir "$TEMP_PATH"/computed
ln -s "$TEMP_PATH"/computed computed
rm -rf unpacked 2>/dev/null || true
