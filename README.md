# Detecting AI Generated Images
This project aims to develop AI models to detect AI Generated images from real images. 

## Setup
1. Set up virtual environment and install dependencies by following commands in terminal.

```
python3.9 -m venv .env
source .env/bin/activate
pip install -r requirement.txt
```

2. Download the datasets from .... and save it in the folder data/raw/stylegan and data/raw/real_images respectively.

3. Build dataset by resizing and splitting into train/dev/test set by running build_dataset.py

```
python build_dataset.py --dataset_dir data/raw/stylegan --output_dir data/64x64/stylegan --image_size 64
python build_dataset.py --dataset_dir data/raw/real --output_dir data/64x64/real --image_size 64
```


## Implementing






