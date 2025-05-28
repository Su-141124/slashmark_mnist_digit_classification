Handwritten Digit Recognition Using CNN (MNIST Dataset):-
This project implements a Convolutional Neural Network (CNN) using TensorFlow and Keras to classify handwritten digits from the MNIST dataset. It also includes functionality for saving/loading the model and making predictions on custom uploaded images.
Overview:-
The MNIST dataset contains 70,000 grayscale images of handwritten digits (0–9), each of size 28x28 pixels. This project trains a CNN to classify these digits and test its accuracy.
Features:-
- Preprocessing and normalization of MNIST data
- Construction of a CNN model with Keras
- Training and evaluation of the model
- Visualization of predictions
- Saving and loading the trained model (mnist_cnn_model.h5)
- Uploading custom images for prediction
- Image preprocessing using OpenCV
Model Architecture:-
- Conv2D: 32 filters, 3x3 kernel, ReLU activation
- MaxPooling2D: 2x2 pooling
- Flatten
- Dense: 64 units, ReLU activation
- Dense: 10 output units, Softmax activation
Requirements:-
- Python 3.x
- TensorFlow
- NumPy
- Matplotlib
- OpenCV (cv2)
- Google Colab (for file upload functionality)
Install the required packages using:
   pip install tensorflow numpy matplotlib opencv-python
Training the Model:
   model.fit(x_train, y_train, epochs=10, validation_split=0.2)
Evaluation:-
After training, the model is evaluated using standard metrics (accuracy), and a few sample predictions are visualized.
Saving and Loading the Model:-
- Save model: model.save("mnist_cnn_model.h5")
- Load model: load_model("mnist_cnn_model.h5")
Predicting Custom Images:-
You can upload your own digit image (in grayscale, resembling MNIST format). The image is resized, inverted, and normalized before making predictions.
