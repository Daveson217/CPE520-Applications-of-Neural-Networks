import torch
import torchvision.transforms as transforms
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load your CNN model class
from .cnn_models import EmotionClassifier, resnet_model

def load_model(model_name:str):
    model = None
    if model_name.strip() == "ours":
        params={
            "dropout_rate": 0.2,
            "num_classes": 7
        }
        model = EmotionClassifier(params)
        model.load_state_dict(torch.load(model.model_path))
        model.to(device)
        model.eval()
    elif model_name.strip() == "resnet":
        return resnet_model()

    return model


def predict_emotion(model, pil_image):
    label_classes = {
        0: 'angry',
        1: 'disgust',
        2: 'fear',
        3: 'happy',
        4: 'neutral',
        5: 'sad',
        6: 'surprise'
    }
    # Model is already loaded

    # Define your transformation
    transform = transforms.Compose([
        transforms.Resize((48, 48)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])  # Adjust based on training
    ])
    image = transform(pil_image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(image)
        _, predicted = outputs.max(1)
    
    return label_classes[predicted.item()]
