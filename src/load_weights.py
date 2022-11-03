import zipfile
import gdown
import os

url = 'https://drive.google.com/u/0/uc?id=1U1YJ3qMGRxsQngkZAntoGE6cu6ahYdZK&export=download'
gdown.download(url, '../weights/weights.zip')
with zipfile.ZipFile('../weights/weights.zip', 'r') as zip_ref:
    zip_ref.extractall('../weights/')
os.remove('../weights/weights.zip')
os.remove('../weights/.gitkeep')

print('Данные загружены')
