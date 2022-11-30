"""This module download test data."""

import os
import zipfile

import gdown

url = "https://drive.google.com/u/0/uc?id=1NiaoxMxe5ckowotiVnRHy0Mk1OWb9cLv&export=download"
gdown.download(url, "../data/test/data.zip")
with zipfile.ZipFile("../data/test/data.zip", "r") as zip_ref:
    zip_ref.extractall("../data/test/")
os.remove("../data/test/data.zip")
os.remove("../data/test/.gitkeep")

print("Данные загружены")
