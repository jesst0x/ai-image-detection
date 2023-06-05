# Detecting AI Generated Images
This project aims to develop AI models to detect AI Generated images from real images. 

## Setup
1. Set up virtual environment and install dependencies by running following commands in terminal.
1. Set up virtual environment and install dependencies by running following commands in terminal.

```
python3.9 -m venv .env
source .env/bin/activate
pip install -r requirement.txt
```


## Preparing Dataset
1. Download datasets listed in reference section. Note that for this project, training sets consists of 7000 images (3500 synthetic and 3500 real). For testing set, each consists of 500 images. 

## Preparing Dataset
1. Download datasets listed in reference section. Note that for this project, training sets consists of 7000 images (3500 synthetic and 3500 real). For testing set, each consists of 500 images. 

2. Running command below in root folder to resize images to desirable size and split the dataset to train/dev/test with ratio 70%/20%/10% 
2. Running command below in root folder to resize images to desirable size and split the dataset to train/dev/test with ratio 70%/20%/10% 

```
python build_dataset.py --dataset_dir data/raw/stylegan --output_dir data/64x64/stylegan --image_size 64


python build_dataset.py --dataset_dir data/raw/real --output_dir data/64x64/real --image_size 64
```

3. We run this command once, to generate resized 64x64 images to be the same input data set for regression model, 2NN and CNN.

4. For CNN, we are training 2 more models with different resolutions, 128x128 and 256 x 256. We can run above command by changing the flag --image_size argument to 128 and 256 respectively to generate correct image size.

5. If we only want to resize testing data set but not splitting for training, just need to add flag ```--is_testing y```.

## Training Model
1. Each model (regression model, 2NN and CNN) has model.py file in model folder, which is main entry point to training the model.
2. In general, to train model, run below command 

```
python model.py --real_img_dir <path to resized real images training dataset> --synthetic_img_dir <path to resized synthetic images training dataset> --logging_dir <path to save training result and parameters>
```
3. There are hyperparameters flags specific to model types that can be specified.
    * ```--epoch```: Number of training epoch
    * ```---learning_rate```: Learning rate for regression model and 2NN
    * ```---hidden_layer```: Number of hidden units in 2NN hidden layer

4. The result and learned parameters is saved in logging directory specified in command above. For this project, it is saved in respective ```experiment``` folder. You may refer to it for a set of already models trained by trying different hyperparameters. 

5. For CNN, trained model is saved in ```trained_model``` folder of logging dir. Due to the size of trained model, it is not uploaded to repo. These trained model can be used to evaluate any test data set.



## Predict and Evaluate on Test Set
1. After training, we can run predication and evaluation on test set by using model trained above.

2. Evaluate by executing ```evaluate.py```

```
python evaluate.py ---model_file <path to trained model> --dataset_dir <path to testing data set> --logging_dir <path to save the test result> --is_synthetic y
```
3. We run this command once, to generate resized 64x64 images to be the same input data set for regression model, 2NN and CNN.

4. For CNN, we are training 2 more models with different resolutions, 128x128 and 256 x 256. We can run above command by changing the flag --image_size argument to 128 and 256 respectively to generate correct image size.

5. If we only want to resize testing data set but not splitting for training, just need to add flag ```--is_testing y```.

## Training Model
1. Each model (regression model, 2NN and CNN) has model.py file in model folder, which is main entry point to training the model.
2. In general, to train model, run below command 

```
python model.py --real_img_dir <path to resized real images training dataset> --synthetic_img_dir <path to resized synthetic images training dataset> --logging_dir <path to save training result and parameters>
```
3. There are hyperparameters flags specific to model types that can be specified.
    * ```--epoch```: Number of training epoch
    * ```---learning_rate```: Learning rate for regression model and 2NN
    * ```---hidden_layer```: Number of hidden units in 2NN hidden layer

4. The result and learned parameters is saved in logging directory specified in command above. For this project, it is saved in respective ```experiment``` folder. You may refer to it for a set of already models trained by trying different hyperparameters. 

5. For CNN, trained model is saved in ```trained_model``` folder of logging dir. Due to the size of trained model, it is not uploaded to repo. These trained model can be used to evaluate any test data set.



## Predict and Evaluate on Test Set
1. After training, we can run predication and evaluation on test set by using model trained above.

2. Evaluate by executing ```evaluate.py```

```
python evaluate.py ---model_file <path to trained model> --dataset_dir <path to testing data set> --logging_dir <path to save the test result> --is_synthetic y
```

3. For this project, the test result has been saved in ```result``` folder by evaluating on different test sets with best trained models. 
3. For this project, the test result has been saved in ```result``` folder by evaluating on different test sets with best trained models. 




## Reference
For some dataset, we only use a subset of those.
1. Training Synthetic Image Dataset (StyleGAN psi0.7): https://drive.google.com/drive/folders/1BGxgDo0qCL8FGYX3Fx1zmO1sYGs-c3Ph
2. Training Real Images Dataset: https://drive.google.com/drive/folders/1tZUcXDBeOibC6jcMCtgRRz67pzrAHeHL
3. Test set (StyleGAN psi0.5): https://drive.google.com/drive/folders/1DWAaNit6yOPGn41psvvEvy5OHyqaHWhp
4. Test set (StyleGAN psi1.0): https://drive.google.com/drive/folders/14uyb1Du_Vc8woAa8Xj9IEFaPGQyl2ptO
5. Test set (ProGAN): https://drive.google.com/drive/folders/1jU-hzyvDZNn_M3ucuvs9xxtJNc9bPLGJ
6. Test set (StyleGAN3): https://nvlabs-fi-cdn.nvidia.com/stylegan3/images/



## Acknowledgement 
The raw dataset used for this project, are provided by 
1. StyleGAN, K. Tero, L. Samuli, A. Timo (2019) _A Style-Based Generator Architecture for
Generative Adversarial Networks_ https://arxiv.org/abs/1812.04948 https://github.com/NVlabs/stylegan
2. StyleGAN3, K. Tero, A. Miika, L. Samuli, H. Erik, H. Janne, L. Jaakko, A. Timo, (2021) _Alias-Free Generative Adversarial Networks_ https://arxiv.org/abs/2106.12423 https://nvlabs.github.io/stylegan3

3. ProGAN, K. Tero, A. Timo, L Samuli, L. Jaakko(2018) _Progressive Growing of GANs for Improved Quality, Stability, and Variation_ https://github.com/tkarras/progressive_growing_of_gans
4. Flickr-Faces-HQ (FFHQ), NVIDIA Corporation,  https://github.com/NVlabs/ffhq-dataset