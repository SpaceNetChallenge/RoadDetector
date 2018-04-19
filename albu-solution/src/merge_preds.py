import os
import tqdm
import numpy as np
import cv2

def merge_tiffs(root):
    os.makedirs(os.path.join(root, 'merged'), exist_ok=True)
    prob_files = {f for f in os.listdir(root) if os.path.splitext(f)[1] in ['.tif', '.tiff']}
    unfolded = {f[6:] for f in prob_files if f.startswith('fold')}
    if not unfolded:
        unfolded = prob_files

    for prob_file in tqdm.tqdm(unfolded):
        probs = []
        for fold in range(4):
            prob = os.path.join(root, 'fold{}_'.format(fold) + prob_file)
            prob_arr = cv2.imread(prob, cv2.IMREAD_GRAYSCALE)
            probs.append(prob_arr)
        prob_arr = np.mean(probs, axis=0)

        res_path_geo = os.path.join(root, 'merged', prob_file)
        cv2.imwrite(res_path_geo, prob_arr)

def merge_tiffs_defferent_folders(roots, res):
    os.makedirs(os.path.join(res), exist_ok=True)
    prob_files = {f for f in os.listdir(roots[0]) if os.path.splitext(f)[1] in ['.tif', '.tiff']}

    for prob_file in tqdm.tqdm(prob_files):
        probs = []
        for root in roots:
            prob_arr = cv2.imread(os.path.join(root, prob_file), cv2.IMREAD_GRAYSCALE)
            probs.append(prob_arr)
        prob_arr = np.mean(probs, axis=0)
        # prob_arr = np.clip(probs[0] * 0.7 + probs[1] * 0.3, 0, 1.)

        res_path_geo = os.path.join(res, prob_file)
        cv2.imwrite(res_path_geo, prob_arr)

if __name__ == "__main__":
    root = '/results/results'
    merge_tiffs(os.path.join(root, '2m_4fold_512_30e_d0.2_g0.2_test'))
