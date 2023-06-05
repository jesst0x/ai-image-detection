import os
import argparse
import random
from PIL import Image
from tqdm import tqdm

def resize(filename, output_dir, image_size=64):
  img = Image.open(filename)
  img = img.resize((image_size, image_size), Image.BILINEAR)
  img.save(os.path.join(output_dir, str(image_size) + 'x' + str(image_size) + '_' + filename.split('/')[-1]))

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_dir', default='data/raw/real', help='Directory with raw dataset')
parser.add_argument('--output_dir', default ='data/64x64/stylegan', help='Directory to save resize dataset')
parser.add_argument('--image_size',
                    default='64',
                    help='Size to resize the image')
parser.add_argument('--test_only', default='n', help='Is the dataset for testing only? y or n')


if __name__ == '__main__':

  args = parser.parse_args()

  # Size to resize image into
  size = int(args.image_size)
  test_only = True if args.test_only == 'y' else False

  data_dir = args.dataset_dir
  images_filenames = [
      os.path.join(data_dir, f) for f in os.listdir(data_dir)
  ]

  random.seed(40)
  random.shuffle(images_filenames)


  # Split into 70% train, 20% development and 10% test set for training data sets. If dataset is only for testing only, it won't be split.
  train_split_ratio = 0 if test_only else 0.7
  dev_split_ratio = 0 if test_only else 0.9
  train_split = int(train_split_ratio * len(images_filenames))
  dev_split = int(dev_split_ratio* len(images_filenames))

  train_filenames = images_filenames[:train_split]
  dev_filenames = images_filenames[train_split:dev_split]
  test_filenames = images_filenames[dev_split:]
  
  splits = {'train': train_filenames, 'dev': dev_filenames, 'test': test_filenames}

  if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir, exist_ok=True)
  
  for d, filenames in splits.items():
    output_dir_split = os.path.join(args.output_dir, d)
    
    if not os.path.exists(output_dir_split):
      os.mkdir(output_dir_split)
    
    for f in tqdm(filenames):
      resize(f, output_dir_split, size)
      
  print('Building dataset is completed')
    
  
  
  
