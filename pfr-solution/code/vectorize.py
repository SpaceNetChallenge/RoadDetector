import rd
import sys

args = sys.argv[1:]
model_root, fold, out_path = args
rd.make_submission(model_root, fold, out_path)
print("done!")
