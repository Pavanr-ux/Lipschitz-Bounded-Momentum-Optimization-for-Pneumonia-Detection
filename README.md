Lipschitz-Bounded Momentum Optimization for Pneumonia Detection
Overview

This project implements a custom optimization technique called Lipschitz-Bounded Momentum Optimization to improve the training stability and convergence of deep learning models.

The method dynamically adjusts the momentum parameter based on the gradient norm, helping to prevent overshooting and improve learning efficiency.

A Convolutional Neural Network (CNN) is used to classify Chest X-ray images into two categories: NORMAL and PNEUMONIA.

Objectives

Develop a CNN model for medical image classification

Implement a custom Lipschitz-based momentum optimizer

Improve convergence stability during training

Compare performance with standard optimizers (Adam)

Deploy the model using a simple web interface

Key Concept

Traditional optimizers use fixed momentum values.
In this project, momentum is dynamically adjusted using the gradient norm.

The idea is:
L equals gradient norm
beta equals beta divided by (1 plus L)

This helps to reduce overshooting when gradients are large and speeds up learning when gradients are small, improving overall stability.

Project Structure

NOP folder contains:
dataset folder (not included in GitHub)
model.py
optimizer.py
utils.py
train.py
compare.py
app.py
requirements.txt
README.md

Technologies Used

Python
PyTorch
NumPy
Matplotlib
Scikit-learn
Gradio

How to Run

Step 1: Install dependencies
Run: python -m pip install -r requirements.txt

Step 2: Train the model
Run: python train.py

Step 3: Run the web application
Run: python app.py

Results

The model shows stable training with decreasing loss across epochs.
It achieves good classification accuracy for pneumonia detection.
The custom optimizer improves convergence compared to traditional methods.

Graphs

Training Loss vs Epochs graph shows steady decrease in loss.
Comparison graph shows improved stability compared to Adam optimizer.

Features

Custom optimizer implementation
Medical image classification
Graphical analysis
Web interface using Gradio

Future Work

Apply the method to larger datasets
Use deeper neural networks like ResNet
Improve Lipschitz estimation
Deploy as a full-scale application

References

PyTorch Documentation
Deep Learning by Ian Goodfellow
Research papers on optimization techniques

Author

Name: Pavan R
Register number: 23BTRCL155
