import matplotlib.pyplot as plt
import numpy as np
import random

def plot_accuracy(history, model_extra_desc):
    """This function plots and saves the training and testing accuracy values"""
        # Ensure tensors are on the CPU and converted to NumPy
    training_accuracy = [t.cpu().numpy() for t in history['training_accuracy']]
    validation_accuracy = [v.cpu().numpy() for v in history['validation_accuracy']]
    
    # Plot accuracy and loss
    plt.figure(figsize=(6, 4))
    plt.plot(training_accuracy, label='Training Accuracy')
    plt.plot(validation_accuracy, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    plt.grid()
    plt.title('Accuracy Over Epochs:' +model_extra_desc)
    plt.savefig('images/' +'Accuracy Over Epochs-' +model_extra_desc + '.png')
    plt.show()
    

def plot_loss(history, model_extra_desc):
    """This function plots and saves the training and testing accuracy values"""
    #training_loss = [t.cpu().numpy() for t in history['training_loss']]
    #validation_loss = [v.cpu().numpy() for v in history['validation_loss']]

    plt.figure(figsize=(6, 4))
    plt.plot(history['training_loss'], label='Training Loss')
    plt.plot(history['validation_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')
    plt.grid()
    plt.title('Loss Over Epochs' +model_extra_desc)
    plt.savefig('images/' +'Loss Over Epochs-' +model_extra_desc + '.png')
    plt.show()
    


def multi_pictured_prediction_plot():
    # Generate predictions for the test images
    predictions = model.predict(test_images)

    # Convert predictions and labels to readable format
    predicted_classes = np.argmax(predictions, axis=1)
    true_classes = np.argmax(test_labels, axis=1)

    # Class labels for visualization
    class_names = {'angry': 0, 'disgust': 1, 'fear': 2, 'happy': 3, 'neutral': 4, 'sad': 5, 'surprise': 6}
    class_names = [key.title() for key in class_names.keys()]

    # Select random test samples to visualize
    num_samples = 12
    sample_indices = random.sample(range(len(test_images)), num_samples)
    sample_images = test_images[sample_indices]
    sample_true_classes = true_classes[sample_indices]
    sample_predicted_classes = predicted_classes[sample_indices]

    # Plot the samples in a grid
    plt.figure(figsize=(15, 10))
    for i, idx in enumerate(sample_indices):
        plt.subplot(3, 4, i + 1)
        plt.imshow(sample_images[i])
        plt.axis('off')
        true_label = class_names[sample_true_classes[i]]
        predicted_label = class_names[sample_predicted_classes[i]]
        color = 'green' if sample_true_classes[i] == sample_predicted_classes[i] else 'orange'
        plt.title(f"True: {true_label}\nPred: {predicted_label}", color=color)

    plt.tight_layout()
    plt.show()
