set -eu
python3 code/do_unpack.py
python3 code/rd.py --provision "$@"
