import os
import shutil
from shutil import copyfile

import pandas as pd
from dotenv import load_dotenv
from utils import make_folder

load_dotenv()
DATA_DIR = os.getenv("DATA_DIR")

#### source data path
s_label = os.path.join(DATA_DIR, "CelebAMaskHQ-mask")
s_img = os.path.join(DATA_DIR, "CelebA-HQ-img")
#### destination training data path
d_train_label = os.path.join(DATA_DIR, "train_label")
d_train_img = os.path.join(DATA_DIR, "train_img")
#### destination testing data path
d_test_label = os.path.join(DATA_DIR, "test_label")
d_test_img = os.path.join(DATA_DIR, "test_img")
#### val data path
d_val_label = os.path.join(DATA_DIR, "val_label")
d_val_img = os.path.join(DATA_DIR, "val_img")

#### make folder
make_folder(d_train_label)
make_folder(d_train_img)
make_folder(d_test_label)
make_folder(d_test_img)
make_folder(d_val_label)
make_folder(d_val_img)

#### calculate data counts in destination folder
train_count = 0
test_count = 0
val_count = 0

image_list = pd.read_csv(
    os.path.join(DATA_DIR, "CelebA-HQ-to-CelebA-mapping.txt"), sep="\s+"
)
f_train = open(os.path.join(DATA_DIR, "train_list.txt"), "w")
f_val = open(os.path.join(DATA_DIR, "val_list.txt"), "w")
f_test = open(os.path.join(DATA_DIR, "test_list.txt"), "w")

for idx, x in enumerate(image_list.iloc[:, 1]):
    print(idx, x)
    if x >= 162771 and x < 182638:
        copyfile(
            os.path.join(s_label, str(idx) + ".png"),
            os.path.join(d_val_label, str(val_count) + ".png"),
        )
        copyfile(
            os.path.join(s_img, str(idx) + ".jpg"),
            os.path.join(d_val_img, str(val_count) + ".jpg"),
        )
        val_count += 1

    elif x >= 182638:
        copyfile(
            os.path.join(s_label, str(idx) + ".png"),
            os.path.join(d_test_label, str(test_count) + ".png"),
        )
        copyfile(
            os.path.join(s_img, str(idx) + ".jpg"),
            os.path.join(d_test_img, str(test_count) + ".jpg"),
        )
        test_count += 1
    else:
        copyfile(
            os.path.join(s_label, str(idx) + ".png"),
            os.path.join(d_train_label, str(train_count) + ".png"),
        )
        copyfile(
            os.path.join(s_img, str(idx) + ".jpg"),
            os.path.join(d_train_img, str(train_count) + ".jpg"),
        )
        train_count += 1

print(train_count + test_count + val_count)
#### close the file
f_train.close()
f_val.close()
f_test.close()
