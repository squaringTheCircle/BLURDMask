import glob
import os

import cv2
import numpy as np

# load data directory
from dotenv import load_dotenv
from utils import make_folder

load_dotenv()
DATA_DIR = os.getenv("DATA_DIR")

label_list = [
    "skin",
    "nose",
    "eye_g",
    "l_eye",
    "r_eye",
    "l_brow",
    "r_brow",
    "l_ear",
    "r_ear",
    "mouth",
    "u_lip",
    "l_lip",
    "hair",
    "hat",
    "ear_r",
    "neck_l",
    "neck",
    "cloth",
]

label_map = {
    "skin": 1,
    "nose": 1,
    "eye_g": 254,
    "l_eye": 2,
    "r_eye": 2,
    "l_brow": 3,
    "r_brow": 3,
    "l_ear": 1,
    "r_ear": 1,
    "mouth": 4,
    "u_lip": 4,
    "l_lip": 4,
    "hair": 5,
    "hat": 254,
    "ear_r": 254,
    "neck_l": 254,
    "neck": 1,
    "cloth": 6,
}

folder_base = os.path.join(DATA_DIR, "CelebAMask-HQ-mask-anno")
folder_save = os.path.join(DATA_DIR, "CelebAMaskHQ-mask")
img_num = 30000

make_folder(folder_save)

for k in range(img_num):
    folder_num = k // 2000
    im_base = np.zeros((512, 512))
    for label in label_list:
        idx = label_map[label]
        filename = os.path.join(
            folder_base, str(folder_num), str(k).rjust(5, "0") + "_" + label + ".png"
        )
        # skip hats, jewerly and glasses
        if idx != 254 and (os.path.exists(filename)):
            print(label, idx)
            im = cv2.imread(filename)
            im = im[:, :, 0]
            im_base[im != 0] = idx

    filename_save = os.path.join(folder_save, str(k) + ".png")
    print(filename_save)
    cv2.imwrite(filename_save, im_base)
