import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
import seaborn as sns

# Load the trained model
model = load_model('asl_optimized_model.h5')

# Prepare ImageDataGenerator for the test set (only rescaling)
test_datagen = ImageDataGenerator(rescale=1.0 / 255)

# Load the test data (make sure you have a separate test dataset)
test_generator = test_datagen.flow_from_directory(
    './data/coco_dataset/valid_classified',  # Path to your test data
    target_size=(224, 224),
    batch_size=32,
    class_mode="categorical",  # Change to "binary" for binary classification
    shuffle=False  # Don't shuffle for evaluation to maintain consistency
)

# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(test_generator)
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")

# Get true labels from the test generator
y_true = test_generator.classes

# Predict the labels using the model
y_pred = np.argmax(model.predict(test_generator), axis=1)

# Generate and print confusion matrix
conf_matrix = confusion_matrix(y_true, y_pred)

# Plot confusion matrix using seaborn
plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=test_generator.class_indices.keys(),
            yticklabels=test_generator.class_indices.keys())
plt.title('Confusion Matrix for ASL Prediction')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

# Generate and print classification report
print("Classification Report:")
print(classification_report(y_true, y_pred, target_names=test_generator.class_indices.keys()))

# Visualize a few test images with predictions
for i in range(5):  # Display 5 random images
    img_path = test_generator.filepaths[i]
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    # Predict the class of the image
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction)
    true_class = y_true[i]

    # Display the image with its predicted and true class
    plt.imshow(img)
    plt.title(f"Predicted: {list(test_generator.class_indices.keys())[predicted_class]}\nTrue: {list(test_generator.class_indices.keys())[true_class]}")
    plt.axis('off')
    plt.show()

# Save the results to a file
with open("test_results.txt", "w") as f:
    f.write(f"Test Loss: {test_loss:.4f}\n")
    f.write(f"Test Accuracy: {test_accuracy:.4f}\n")
    f.write(f"Confusion Matrix:\n{conf_matrix}\n")
    f.write(f"Classification Report:\n{classification_report(y_true, y_pred, target_names=test_generator.class_indices.keys())}\n")

