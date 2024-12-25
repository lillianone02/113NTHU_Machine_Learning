# Topic: Facial Expression Recognition

## Description
You can use the 5 files to build and implementation your own **Facial Expression Recognition model**.
- `landmark.py`: This program can **detect human face** from an image, and also **landmard facial features** by 68 green points.
- `data_preprocessing`: This program can do **data augmentation**, and others preprocessing procesure.
- `models.py`: This program have several **cnn models**. You can choose one you like.
- `train.py`: This program build and implement the whole model **training and testing process**.
- `best_model_fold5.pth`: This is the **weights of the best trained model**. You can use the weights instead of trained by yourself.

## Environment
- Pytorch
- Python 3.10
- Dlib
- Mtcnn
- Virtual environment

## How to use the program

### Use the pre-trained model

1. Download 5 files and put them under `a folder`.
2. Put your data folder under `the same folder` as step 1.
3. Run `run_model1`

### Train model by yourself
1. Download 5 files and put them under `a folder`.
2. Put your data folder under `the same folder` as step 1.
3. Run `landmark.py`.
4. Run `data_preprocessing`.
5. Run `models.py`.
6. Run `train.py`.

## Dataset
If you want to use AffectNet, you have to access data from te author from [HERE](http://mohammadmahoor.com/affectnet-request-form/).

You can also use your own data.
