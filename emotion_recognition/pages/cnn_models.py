from pathlib import Path

import torch
from torch.autograd import Variable
import torch.nn as nn
from torchvision import models

BASE_DIR = Path(__file__).resolve().parent.parent.parent
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Our Model with SGD
class EmotionClassifier(nn.Module):
    def __init__(self, params):
        super(EmotionClassifier, self).__init__()        
        num_classes = params['num_classes']
        self.model_path = Path.joinpath(BASE_DIR, 'best_weights_our_cnn_SGD.pt')

        # First convolutional layer
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2) # Output: 32x24x24
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.layer4 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.layer5 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            #nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Fully connected (dense) layers
        self.fc1 = nn.Linear(in_features=512*3*3, out_features=1024)
        self.dropout = nn.Dropout2d(params['dropout_rate'])
        self.fc2 = nn.Linear(in_features=1024, out_features=256)
        self.fc3 = nn.Linear(in_features=256, out_features=num_classes)
       
        #self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        output = self.layer1(x)  # Pass through first layer
        output = self.layer2(output)  # Pass through second layer
        output = self.layer3(output)  # Pass through third layer
        output = self.layer4(output)  # Pass through fourth layer
        output = self.layer5(output)  # Pass through fifth layer
        output = output.view(output.size(0), -1)  # Flatten for the fully connected layers
        output = self.fc1(output)  # First fully connected layer
        output = self.dropout(output)  # Apply dropout
        output = self.fc2(output)  # Second fully connected layer
        output = self.fc3(output)  # Output layer
        
        return output
        

def resnet_model():
    resnet_path  = Path.joinpath(BASE_DIR, 'best_weights_resnet_sgd.pt')
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT) # Load pretrained ResNet-50    
    # Replace the last fully connected layer
    model.fc = nn.Linear(model.fc.in_features, 7)
    model.load_state_dict(torch.load(resnet_path))
    model.to(device)
    return model
