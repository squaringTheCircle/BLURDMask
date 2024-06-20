import json
import os

import numpy as np
from dotenv import find_dotenv, load_dotenv
from PIL import Image
from skimage import morphology as mp

load_dotenv(find_dotenv("../.env"))

LOAD_MASK = ["face", "eyes", "eyebrows", "lips", "hair", "shirt", "tie"]
BLURD_DIR = os.getenv("BLURD_DIR")
RESIZE = 1024


def load_mask(masks_dir, mask_fname):
    return np.array(Image.open(os.path.join(masks_dir, mask_fname)), dtype=bool)


def get_fname(fnames, tag):
    return [m_fn for m_fn in fnames if tag in m_fn][0]


with open(
    os.path.join(BLURD_DIR, "rendered_dataset.json"), "r", encoding="utf-8"
) as file:
    json_data = json.load(file)

for uuid, factors in json_data.items():
    gender = factors["gender"]
    uuid_pth = os.path.join(BLURD_DIR, gender, uuid)
    masks = []
    masks_dir = os.path.join(uuid_pth, "masks")
    mask_fnames = os.listdir(masks_dir)

    # get Skin mask
    face_mask_fname = get_fname(mask_fnames, "face_")
    face_mask = load_mask(masks_dir, face_mask_fname)

    eyelashes_mask_fname = get_fname(mask_fnames, "eyelashes_")
    eyelashes_mask = load_mask(masks_dir, eyelashes_mask_fname)
    eyelashes_mask = mp.binary_erosion(
        eyelashes_mask, footprint=mp.diamond(3, decomposition="sequence")
    )

    beard_mask_fname = get_fname(mask_fnames, "beard_")
    beard_mask = load_mask(masks_dir, beard_mask_fname)
    beard_mask = mp.binary_erosion(
        beard_mask, footprint=mp.diamond(3, decomposition="sequence")
    )
    beard_mask = mp.remove_small_holes(beard_mask, area_threshold=int(64 * 3))
    beard_mask = mp.remove_small_objects(beard_mask, connectivity=2)

    face_mask = (face_mask | eyelashes_mask) | beard_mask

    face_mask = mp.binary_dilation(face_mask)
    face_mask = mp.remove_small_holes(face_mask, area_threshold=int(64 * 3))
    skin_mask = mp.binary_erosion(face_mask)

    # get eyes mask
    eyes_mask_fname = get_fname(mask_fnames, "eyes_")
    eyes_mask = load_mask(masks_dir, eyes_mask_fname)
    eyes_mask = mp.binary_dilation(eyes_mask)
    eyes_mask = mp.remove_small_holes(eyes_mask)

    # get eyebrows mask
    eyebrows_mask_fname = get_fname(mask_fnames, "eyebrows_")
    eyebrows_mask = load_mask(masks_dir, eyebrows_mask_fname)
    eyebrows_mask = mp.binary_dilation(
        eyebrows_mask, footprint=mp.diamond(5, decomposition="sequence")
    )
    eyebrows_mask = mp.remove_small_holes(eyebrows_mask, area_threshold=int(64))
    eyebrows_mask = mp.binary_erosion(
        eyebrows_mask, footprint=mp.diamond(5, decomposition="sequence")
    )

    # get lips mask
    lips_mask_fname = get_fname(mask_fnames, "lips_")
    lips_mask = load_mask(masks_dir, lips_mask_fname)

    # get hair mask
    hair_mask_fname = get_fname(mask_fnames, "hair_")
    hair_mask = load_mask(masks_dir, hair_mask_fname)
    hair_mask = mp.binary_dilation(
        hair_mask, footprint=mp.diamond(3, decomposition="sequence")
    )
    hair_mask = mp.remove_small_holes(hair_mask, area_threshold=int(64 * 5))
    hair_mask = mp.binary_erosion(
        hair_mask, footprint=mp.diamond(3, decomposition="sequence")
    )

    # get cloth mask
    shirt_mask_fname = get_fname(mask_fnames, "shirt_")
    shirt_mask = load_mask(masks_dir, shirt_mask_fname)

    tie_mask_fname = get_fname(mask_fnames, "tie")
    tie_mask = load_mask(masks_dir, tie_mask_fname)

    cloth_mask = shirt_mask | tie_mask

    masks = {
        "skin": (1, skin_mask),
        "eyebrows": (3, eyebrows_mask),
        "hair": (5, hair_mask),
        "eyes": (2, eyes_mask),
        "lips": (4, lips_mask),
        "cloth": (6, cloth_mask),
    }
    mask_order = ["skin", "eyebrows", "hair", "eyes", "lips", "cloth"]

    label = np.zeros((RESIZE, RESIZE), dtype=np.uint8)
    for m in mask_order:
        value, mask_arr = masks[m]
        mask_img = Image.fromarray(mask_arr)
        mask_img = mask_img.resize((RESIZE, RESIZE), resample=Image.Resampling.NEAREST)
        mask_arr = np.array(mask_img)
        label[mask_arr] = value

    label_img = Image.fromarray(label)

    save_pth = os.path.join(uuid_pth, "label")
    os.makedirs(save_pth, exist_ok=True)
    save_fname = os.path.join(save_pth, "label.png")
    label_img.save(save_fname)
