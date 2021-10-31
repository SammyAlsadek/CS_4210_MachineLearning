# -------------------------------------------------------------------------
# AUTHOR: Sammy Alsadek
# FILENAME: perceptron.py
# SPECIFICATION:
#  Python program (perceptron.py) that will read the file optdigits.trato build a Single  Layer Perceptron
#  and a Multi-Layer  Perceptron classifiers. You  will  simulate  a  grid  search, trying tofind which
#  combination of twohyperparameters (learning rate andshuffle) leads you to the best prediction performancefor
#  each classifier. To test the accuracy of those distinct models, you will use the file optdigits.tes.
#  You should update and print the accuracyof each classifier, together with the hyperparameters when it is
#  getting higher
# FOR: CS 4210- Assignment #4
# TIME SPENT: 45 minutes
# -----------------------------------------------------------*/

# IMPORTANT NOTE: YOU HAVE TO WORK WITH THE PYTHON LIBRARIES numpy AND pandas to complete this code.

# importing some Python libraries
from sklearn.linear_model import Perceptron
# pip install scikit-learn==0.18.rc2 if needed
from sklearn.neural_network import MLPClassifier
import numpy as np
import pandas as pd

n = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0]
r = [True, False]

# reading the data by using Pandas library
df = pd.read_csv('optdigits.tra', sep=',', header=None)

# getting the first 64 fields to form the feature data for training
X_training = np.array(df.values)[:, :64]
# getting the last field to form the class label for training
y_training = np.array(df.values)[:, -1]

# reading the data by using Pandas library
df = pd.read_csv('optdigits.tes', sep=',', header=None)

# getting the first 64 fields to form the feature data for test
X_test = np.array(df.values)[:, :64]
# getting the last field to form the class label for test
y_test = np.array(df.values)[:, -1]

highPerAcc = 0
highMLPAcc = 0

for w in n:  # iterates over n

   for b in r:  # iterates over r

      for a in range(2):  # iterates over the algorithms

         # Create a Neural Network classifier
         if a == 0:
            # eta0 = learning rate, shuffle = shuffle the training data
            clf = Perceptron(eta0=w, shuffle=b, max_iter=1000)
         else:
            # learning_rate_init = learning rate, hidden_layer_sizes = number of neurons in the ith hidden layer, shuffle = shuffle the training data
            clf = MLPClassifier(activation='logistic', learning_rate_init=w, hidden_layer_sizes=(
               25,), shuffle=b, max_iter=1000)

         # Fit the Neural Network to the training data
         clf.fit(X_training, y_training)

         # make the classifier prediction for each test sample and start computing its accuracy
         # hint: to iterate over two collections simultaneously with zip() Example:
         # for (x_testSample, y_testSample) in zip(X_test, y_test):
         # to make a prediction do: clf.predict([x_testSample])
         # --> add your Python code here
         correctCount = 0
         for i in range(len(X_test)):
            prediction = clf.predict([X_test[i]])
            if prediction == y_test[i]:
               correctCount += 1
         accuracy = correctCount/len(X_test)

         # check if the calculated accuracy is higher than the previously one calculated for each classifier. If so, update the highest accuracy and print it together with the network hyperparameters
         #Example: "Highest Perceptron accuracy so far: 0.88, Parameters: learning rate=0.01, shuffle=True"
         #Example: "Highest MLP accuracy so far: 0.90, Parameters: learning rate=0.02, shuffle=False"
         # --> add your Python code here
         if a==0:
            if accuracy > highPerAcc:
               highPerAcc = accuracy
               print("Highest Perceptron accuracy so far: " + str(highPerAcc) + ", Parameters: learning rate=" + str(w) + ", shuffle=" + str(b))
         else:
            if accuracy > highMLPAcc:
               highMLPAcc = accuracy
               print("Highest MLP accuracy so far: " + str(highMLPAcc) + ", Parameters: learning rate=" + str(w) + ", shuffle=" + str(b))
