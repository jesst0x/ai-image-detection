import os
import argparse
import random
from PIL import Image
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_dir', default='data/raw/stylegan', help='Directory with raw dataset')
parser.add_argument('--output_dir', default ='data/64x64/stylegan', help='Directory to save resize dataset')
parser.add_argument('--image_size',
                    default='64',
                    help='Size to resize the image')

parser.add_argument('--data_size', default='5000', help='Number of examples to extract')

if __name__ == '__main__':

  args = parser.parse_args()

  # Size to resize image into
  size = int(args.image_size)
  example_count = int(args.data_size)

  data_dir = args.dataset_dir
  images_filenames = [
      os.path.join(data_dir, f) for f in os.listdir(data_dir)
  ]

  random.seed(40)
  random.shuffle(images_filenames)

  images_filenames = images_filenames[:example_count]
  # Split into 70% train, 20% development and 10% test set
  train_split = int(0.7 * len(images_filenames))
  dev_split = int(0.9 * len(images_filenames))

  train_filenames = images_filenames[:train_split]
  dev_filenames = images_filenames[train_split:dev_split]
  test_filenames = images_filenames[dev_split:]
  
  splits = {'train': train_filenames, 'dev': dev_filenames, 'test': test_filenames}
  
  def resize(filename, output_dir, image_size=64):
    img = Image.open(filename)
    img = img.resize((image_size, image_size), Image.BILINEAR)
    img.save(os.path.join(output_dir, filename.split('/')[-1]))

  if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir, exist_ok=True)
  
  for d, filenames in splits.items():
    output_dir_split = os.path.join(args.output_dir, d)
    
    if not os.path.exists(output_dir_split):
      os.mkdir(output_dir_split)
    
    for f in tqdm(filenames):
      resize(f, output_dir_split, size)
      
  print('Building dataset is completed')
    
  
  
  
