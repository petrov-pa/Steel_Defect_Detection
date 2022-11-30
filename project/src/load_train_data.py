"""This module download train data."""

import os
import zipfile
import gdown


url = "https://drive.google.com/u/0/uc?id=1l69U8oY90CTC3c04dNvz_XcBk5-RRohO&export=download"
gdown.download(url, "../data/train/data.zip")
with zipfile.ZipFile("../data/train/data.zip", "r") as zip_ref:
    zip_ref.extractall("../data/train/")
os.remove("../data/train/data.zip")
os.remove("../data/train/.gitkeep")

print("Данные загружены")
