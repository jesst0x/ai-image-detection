import util
import argparse
import os
import json
import tensorflow as tf

from tensorflow.keras import layers, models
# import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--epoch', default ='10', help='Directory to save resize dataset')
parser.add_argument('--logging_dir', default ='../experiment/epoch_5_128x128', help='Directory to save experiment result')
parser.add_argument('--real_img_dir', default ='../../data/128x128/real', help='Directory to real image dataset')
parser.add_argument('--synthetic_img_dir', default ='../../data/128x128/stylegan', help='Directory to synthetic image dataset')
parser.add_argument('--image_size', default ='128', help='Input image size')

# Main training model
def model(train_x, train_y, dev_x, dev_y, num_epoch, logging_dir, image_size):
    
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(image_size, image_size, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))
      
    with open(os.path.join(logging_dir, 'model_layers.txt'), 'w') as f:
        model.summary(print_fn=lambda x: f.write(x + '\n'))
    
    # checkpoint directory
    checkpoint_dir = os.path.join(logging_dir, 'checkpoint')
    if not os.path.exists(checkpoint_dir):   
        os.mkdir(checkpoint_dir)
    checkpoint_path = os.path.join(checkpoint_dir, 'trained_model.ckpt')

    # Save the trained weights using the `checkpoint_path` format
    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path, 
        verbose=1, 
        save_weights_only=True, save_freq=219 * 2)  
    model.save_weights(checkpoint_path.format(epoch=0))
    
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    history = model.fit(train_x, train_y, epochs=num_epoch, callbacks=[cp_callback],validation_data=(dev_x, dev_y), shuffle=True)
    
    train_loss, train_acc = model.evaluate(train_x,  train_y, verbose=2)
    dev_loss, dev_acc = model.evaluate(dev_x,  dev_y, verbose=2)
    print(train_loss, train_acc,dev_loss, dev_acc)
    
    # Save trained model and result of training
    model.save(os.path.join(logging_dir, 'trained_model'))
    json.dump(history.history, open(os.path.join(logging_dir, 'result.json'), 'w'))
    
    # Plot history
    util.plot_history(history.history, logging_dir)


if __name__ == '__main__':
    args = parser.parse_args()
    num_epoch = int(args.epoch)
    logging_dir = args.logging_dir
    real_img_dir = args.real_img_dir
    synthetic_img_dir = args.synthetic_img_dir
    image_size = int(args.image_size)

    # Make directory to save our experimental parameters
    logging_dir =  os.path.join('../experiment', logging_dir)
    if not os.path.exists(logging_dir):   
        os.mkdir(logging_dir)
    else:
        print('Logging directory already exists, overwritting result')
    
        
    # Load dataset with features extracted
    train_x, train_y, dev_x, dev_y = util.load_training_dataset(synthetic_img_dir, real_img_dir)
    
    # Running model and evaluate
    model(train_x, train_y, dev_x, dev_y, num_epoch, logging_dir, image_size)
  
    