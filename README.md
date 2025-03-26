# Facial Recognition of Emotion with Convolutional Neural Networks: a comparison of different models

## Abstract
Facial emotion recognition applications range from human-computer interactions to healthcare. In light of this, we proposed a CNN for emotion recognition in this work evaluated against ResNet50, VGG, and GoogLeNet with a dataset of grayscale 48x48 facial images of seven classes of emotion. We implemented normalization, flipping,
and affine transformation as steps for preprocessing.

Our proposed model achieved 74% training accuracy and 65% test accuracy on the validation set trained using SGD, outperforming its Adam counterpart, while ResNet50 scored 97% and 68% on training and
testing, respectively. Live testing was implemented via a Django app with OpenCV, which pointed out some challenges. Misclassification of the Disgust class arising from data imbalance was noted among other issues. Results were good, but also underlined the necessity of addressing FER's class imbalance issues.

- You need to download the dataset [https://www.kaggle.com/datasets/jonathanoheix/face-expression-recognition-dataset](here) and extract the contents into a folder named `jonathan_oheix`.
- Create a virtual environment with Anaconda or venv: `python -m venv fervenv`. Ensure the environment is activated.
- Run `pip install requirements.txt`

## Demo App (Django)
- To set up the app for the demo app, you need to run `python manage.py makemigrations` then `python manage.py migrate`.
- Run `python manage.py runserver` to start the server.
