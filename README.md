# Facial Recognition of Emotion with Convolutional Neural Networks: a comparison of different models
> This was a group project I did with Samuel Olugbemi.

## Abstract
Facial emotion recognition applications range from human-computer interactions to healthcare. In light of this, we proposed a CNN for emotion recognition in this work evaluated against ResNet50, VGG, and GoogLeNet with a dataset of grayscale 48x48 facial images of seven classes of emotion. We implemented normalization, flipping,
and affine transformation as steps for preprocessing.

Our proposed model achieved 74% training accuracy and 65% test accuracy on the validation set trained using SGD, outperforming its Adam counterpart, while ResNet50 scored 97% and 68% on training and
testing, respectively. Live testing was implemented via a Django app with OpenCV, which pointed out some challenges. Misclassification of the Disgust class arising from data imbalance was noted among other issues. Results were good, but also underlined the necessity of addressing FER's class imbalance issues.

## Setting up code
- You need to download the dataset [https://www.kaggle.com/datasets/jonathanoheix/face-expression-recognition-dataset](here) and extract the contents into a folder named `jonathan_oheix`.
- Create a virtual environment with Anaconda or venv: `python -m venv fervenv`. Ensure the environment is activated.
- Run `pip install requirements.txt`
- You can then use any of the Jupyter notebook files to train your model.
- In the Jupyter notebooks, you may need to modify *emotion_recognition/pages/cnn_models.py/EmotionClassifier* class (Line 16) to include the name of your saved model weights. It has to be in the root directory just as it is on GitHub.
- You also need to modify *emotion_recognition/pages/cnn_models.py/resnet_model* with the appropriate path as well.
- We have uploaded our saved weights. You can use that as well.

## Demo App (Django)
- To set up the app for the demo app, you need to run `python manage.py makemigrations` then `python manage.py migrate`.
- Run `python manage.py runserver` to start the server.

![image](https://github.com/user-attachments/assets/9c9ff373-dc7d-4e00-9907-fa40b186eb53)

![image](https://github.com/user-attachments/assets/154d344a-002a-48cb-b6e0-907f823b31db)


|     |  |
| -------- | ------- |
| ![image](https://github.com/user-attachments/assets/64f7ff66-8acb-4cc3-bc93-8140ad24046a)| ![image](https://github.com/user-attachments/assets/0bf27905-2eb7-428d-9931-34e08065b50e) |
| ![image](https://github.com/user-attachments/assets/6c23cd51-a9b1-4c96-85be-c95d746347ec)| ![image](https://github.com/user-attachments/assets/b557ef19-ef42-4f38-a6f0-e144207a514e) |



