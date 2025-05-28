import tensorflow as tf

# Load the dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Normalize pixel values (scale between 0 and 1)
x_train, x_test = x_train / 255.0, x_test / 255.0

# Print dataset shapes
print(f"Training data shape: {x_train.shape}, Labels: {y_train.shape}")
print(f"Test data shape: {x_test.shape}, Labels: {y_test.shape}")

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# Define the CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')  # 10 output classes (digits 0-9)
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Show model summary
model.summary()

history = model.fit(x_train, y_train, epochs=10, validation_split=0.2)

import numpy as np

# Make predictions
predictions = model.predict(x_test)

# Convert predictions to class labels (0-9)
predicted_labels = np.argmax(predictions, axis=1)

# Print first 10 predictions
print("Predicted labels:", predicted_labels[:10])
print("Actual labels:   ", y_test[:10])

import matplotlib.pyplot as plt

# Plot the first 10 test images and their predictions
plt.figure(figsize=(10, 5))

for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(x_test[i], cmap="gray")  # Show image in grayscale
    plt.title(f"Pred: {predicted_labels[i]}\nActual: {y_test[i]}")
    plt.axis("off")

plt.tight_layout()
plt.show()

index = 5  # Change this to test a different image
plt.imshow(x_test[index], cmap="gray")
plt.title(f"Predicted: {predicted_labels[index]}, Actual: {y_test[index]}")
plt.axis("off")
plt.show()

# Save the trained model
model.save("mnist_cnn_model.h5")
print("Model saved successfully!")

from google.colab import files
files.download("mnist_cnn_model.h5")

from tensorflow.keras.models import load_model

# Load the saved model
loaded_model = load_model("mnist_cnn_model.h5")
print("Model loaded successfully!")

from google.colab import files
uploaded = files.upload()

import cv2
import numpy as np
import matplotlib.pyplot as plt

# Get the uploaded file name
if uploaded:  # Check if there are uploaded files
    # Get the uploaded file name (assuming only one file is uploaded)
    uploaded_file_name = next(iter(uploaded))

    # Load the uploaded image
    image_path = uploaded_file_name  # Use the file name
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Load in grayscale

    # Resize to 28x28 (same as MNIST images)
    image = cv2.resize(image, (28, 28))

    # Invert colors (MNIST background is black, but your image might be white)
    image = cv2.bitwise_not(image)

    # Normalize pixel values (0 to 1)
    image = image / 255.0

    # Reshape to match model input shape (28, 28, 1)
    image = image.reshape(1, 28, 28, 1)

    # Show the image
    plt.imshow(image[0], cmap="gray")
    plt.title("Processed Image")
    plt.axis("off")
    plt.show()

else:
    print("No file uploaded. Please upload an image file.")

# Make prediction
prediction = loaded_model.predict(image)
predicted_label = np.argmax(prediction)

# Show the prediction
print(f"Predicted Digit: {predicted_label}")

