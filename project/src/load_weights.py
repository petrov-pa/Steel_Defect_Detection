"""This module download weights of models."""

import os
import zipfile

import gdown


url = "https://drive.google.com/uc?export=download&confirm=no_antivirus&id=1U1YJ3qMGRxsQngkZAntoGE6cu6ahYdZK"
gdown.download(url, "../weights/weights.zip")
with zipfile.ZipFile("../weights/weights.zip", "r") as zip_ref:
    zip_ref.extractall("../weights/")
os.remove("../weights/weights.zip")
os.remove("../weights/.gitkeep")

print("Данные загружены")
