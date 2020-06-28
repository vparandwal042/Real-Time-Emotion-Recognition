# Real-Time-Emotion-Recognition
A system that predicts or recognize seven basic human emotions (Happy, Angry, Sad, Fear, Disgust, Surprise and Neutral). For user convenience Web-App and Desktop-App is developed.

## Emotion Recognition
Used Convolutional neural networks (CNN) for facial expression recognition. The goal is to classify each facial image into one of the seven facial emotion categories considered. This Project has two applications:<br>
&emsp;1. Desktop Application (using PyQt5)<br>
&emsp;2. Web Application (using Flask)

## Data
We trained and tested our models on the data set from the Kaggle Facial Expression Recognition Challenge (https://www.kaggle.com/c/challenges-in-representation learning-facial-expression-recognition-challenge), which comprises 48-by-48-pixel grayscale images of human faces, each labelled with one of 7 emotion categories: anger, disgust, fear, happiness, sadness, surprise, and neutral.<br>
Image set of 35,887 examples, with training-set: validation-set: test-set as 80: 10: 10.

## Prerequisites
&emsp;pip install tensorFlow<br>
&emsp;pip install keras<br>
&emsp;pip install numpy<br>
&emsp;pip install sklearn<br>
&emsp;pip install pandas<br>
&emsp;pip install scikitplot<br>
&emsp;pip install opencv-python<br>
&emsp;pip install matplotlib<br>
&emsp;pip install pyqt5-python<br>
&emsp;pip install Flask<br>

## Pre-Processing
Pre-processing of the FER2013 dataset just extracts valuable or relevant images (pixel values). This data set contains some missing values, irrelevant values, etc. This can reduce the accuracy of any model. Thus, pre-processing is the first step for increased accuracy for the proposed method.

## Getting started
1. Download FER2013 dataset from Kaggle Facial Expression Recognition Challenge (https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge) and extract in the main folder.

2. Training:-<br> 
&emsp;a)To train deep CNN model. Open the terminal and navigate to the project folder and run cnn_deep.py file.<br>
&emsp;&emsp;&emsp;python cnn_deep.py<br>
&emsp;&emsp;No need to train the model, already trained weights saved in model4layer_2_2_pool.h5 file.<br><br>
&emsp;b)To train the shallow CNN model. Open the terminal and navigate to the project folder and run cnn_shallow.py file.<br>
&emsp;&emsp;&emsp;python cnn_shallow.py<br>
&emsp;&emsp;No need to train the model, already trained weights saved in model2layer_2_2_pool.h5 file.

3. Want to train the model yourself?<br>
&emsp;Just change the statement is_model_saved = True to is_model_saved = False.

4. To test the trained model run RTFE_test.py, to get accuracy on the testing dataset.<br>
&emsp;&emsp;&emsp;python RTFE_test.py

5. If you want to test on images run RTFE_testcustom.py.<br>
&emsp;&emsp;&emsp;python RTFE_testcustom.py

6. For Real-Time Emotion Recognition:-<br> 
&emsp;a)run Desktop App Emotion_Recognition.py<br>
&emsp;&emsp;&emsp;python Emotion_Recognition.py<br><br>
&emsp;b)run Web App app.py<br>
&emsp;&emsp;&emsp;python app.py<br>
&emsp;&emsp;after this open browser and enter URL- "http:127.0.0.1:5000". This will open Web-App.

## Model Training

**Shallow Convolutional Neural Network (SHCNN)**<br>
&emsp;First, we built a shallow CNN. This network had two convolutional layers and one FC layer.<br>
&emsp;First convolutional layer, we had 64 3×3 filters, along with batch normalization and dropout and max-pooling with a filter size 2×2.<br>
&emsp;Second convolutional layer, we had 128 5×5 filters, along with batch normalization and dropout and max-pooling with a filter size 2×2.<br>
&emsp;In the FC layer, we had a hidden layer with 256 neurons and Sigmoid as the loss function.<br><br>
**Deep Convolutional Neural Network (DCNN)**<br>
&emsp;A. To improve accuracy we used deeper CNN. This network had 4 convolutional layers and with 2 FC layer.<br>
&emsp;&emsp;The first layer of convolution uses 64 5×5 convolution kernels, while the second convolution layer, the third convolution layer and fourth convolution layer use the same size of the kernel, i.e. 128, 512, and 512 3×3 convolution kernel.<br><br>
&emsp;&emsp;The ReLU layer that applying the ReLU activation function such as max (0, x) is used at each layer after each convolution. Max-pooling layer of 2×2 is embedded into each layer after ReLU layer.<br><br>
&emsp;&emsp;As the Fully Connected layer is Flattening of image matrix that is the output of previous three layers, the three layers (i.e. Are in matrix form) are converted into a vector and feed into first fully connected layer which containing 256 neurons then output if first fully connected layer is fed into a second fully connected layer which contains 512 neurons.<br><br>
&emsp;B. Using Google Colab the training time rapidly reduced. So, accuracy increases. DCNN architecture is the same as above.

## Optimizers and Loss Function

**Adam**<br>
&emsp;Adam can be looked at as a combination of RMSprop and Stochastic Gradient Descent with momentum. It uses the squared gradients to scale the learning rate like RMSprop and it takes advantage of momentum by using moving average of the gradient instead of the gradient itself like SGD with momentum.<br><br>
**Loss Function**<br>
&emsp;The goal of machine learning and deep learning is to reduce the difference between the predicted output and the actual output. This is also called as a Cost function (C) or Loss function. We have used binary cross-entropy loss function (BCE). BCE loss is used for the binary classification tasks. BCE Loss creates a criterion that measures the Binary Cross Entropy between the target and the output. If we use the BCE Loss function, we need to have a sigmoid layer in our network.

## Accuracy Achieved

&emsp;Shallow CNN -- 75.59%<br>
&emsp;Deep-CNN    -- 78.29%<br>
&emsp;Deep-CNN using Google Colab -- 90.95%
