**🧠 Handwritten Digit Classification using Neural Network**

This project demonstrates how to build and train a Neural Network model to classify handwritten digits (0–9) using the MNIST dataset. The implementation is done in Python using TensorFlow/Keras and NumPy, and the complete workflow is provided in a Jupyter Notebook.

**📌 Project Overview**

Handwritten digit recognition is a classic problem in Computer Vision and Machine Learning.
In this project, a neural network is trained to accurately recognize digits from grayscale images of size 28×28 pixels.

**🚀 Features**

Uses the MNIST handwritten digit dataset

Data preprocessing and normalization

Neural Network model built using Keras

Model training with validation

Accuracy and loss evaluation

Prediction on test images

Simple and beginner-friendly implementation

🛠️ Technologies Used

Python 3

TensorFlow / Keras

NumPy

Matplotlib

Jupyter Notebook

**📂 Project Structure**
Handwritten-Digit-Classification/
│
├── Handwritten_digit_classification_using_neural_network.ipynb
├── README.md

**📊 Dataset**

MNIST Dataset

60,000 training images

10,000 testing images

Each image is 28×28 pixels

Digits range from 0 to 9

The dataset is automatically loaded using keras.datasets.mnist.

⚙️ Installation & Setup

Clone the repository

git clone https://github.com/your-username/handwritten-digit-classification.git
cd handwritten-digit-classification


Install required packages

pip install tensorflow numpy matplotlib


Run the notebook

jupyter notebook

Open Handwritten_digit_classification_using_neural_network.ipynb.

**🧪 Model Architecture**

Input layer (Flatten 28×28 images)

Hidden Dense layers with ReLU activation

Output Dense layer with Softmax activation
Optimizer: Adam

Loss function: Sparse Categorical Crossentropy

**📈 Results**

High training and validation accuracy

Low loss after training

Correct predictions for most handwritten digits

**🔍 Sample Prediction**

The model can predict a digit from a test image and display:

Input image

Actual label

Predicted label
