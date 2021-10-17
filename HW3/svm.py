# -------------------------------------------------------------------------
# AUTHOR: Sammy Alsadek
# FILENAME: svm.py
# SPECIFICATION:
#   Complete the Python program (svm.py) that will also read the file optdigits.tra
#   to build multiple SVM classifiers.You will simulate a grid search, trying to find
#   which combination of four SVM hyperparameters
#   (c, degree, kernel, and decision_function_shape) leads you to the best prediction
#   performance. To test the accuracy of those distinct models, you will also use the
#   file optdigits.tes.You should update and print the accuracy, together with the
#   hyperparameters, when it is getting higher.
# FOR: CS 4210- Assignment #3
# TIME SPENT: 45 minutes
# -----------------------------------------------------------*/

# IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH AS numpy OR pandas. You have to work here only with standard vectors and arrays

# importing some Python libraries
from sklearn import svm
import csv

dbTest = []
X_training = []
Y_training = []
c = [1, 5, 10, 100]
degree = [1, 2, 3]
kernel = ["linear", "poly", "rbf"]
decision_function_shape = ["ovo", "ovr"]
highestAccuracy = 0

# reading the data in a csv file
with open('optdigits.tra', 'r') as trainingFile:
  reader = csv.reader(trainingFile)
  for i, row in enumerate(reader):
      X_training.append(row[:-1])
      Y_training.append(row[-1])

# reading the data in a csv file
with open('optdigits.tes', 'r') as testingFile:
  reader = csv.reader(testingFile)
  for i, row in enumerate(reader):
      dbTest.append(row)

# created 4 nested for loops that will iterate through the values of c, degree, kernel, and decision_function_shape
# --> add your Python code here

for elementInC in c:  # iterates over c
    for elementInDegree in degree:  # iterates over degree
        for elementInKernel in kernel:  # iterates kernel
           for elementInDFS in decision_function_shape:  # iterates over decision_function_shape

                # Create an SVM classifier that will test all combinations of c, degree, kernel, and decision_function_shape as hyperparameters. For instance svm.SVC(c=1)
                clf = svm.SVC(C=elementInC, degree=elementInDegree, kernel=elementInKernel, decision_function_shape=elementInDFS)

                # Fit SVM to the training data
                clf.fit(X_training, Y_training)

                # make the classifier prediction for each test sample and start computing its accuracy
                # --> add your Python code here
                correct_guess = 0
                for testSample in dbTest:
                    class_predicted = int(clf.predict([testSample[:len(testSample)-1]])[0])
                    correct_guess += 1 * (class_predicted == int(testSample[len(testSample)-1]))
                accuracy = correct_guess/len(dbTest)

                # check if the calculated accuracy is higher than the previously one calculated. If so, update update the highest accuracy and print it together with the SVM hyperparameters
                # Example: "Highest SVM accuracy so far: 0.92, Parameters: a=1, degree=2, kernel= poly, decision_function_shape = 'ovo'"
                # --> add your Python code here
                if accuracy > highestAccuracy:
                    highestAccuracy = accuracy
                    print("Highest SVM accuracy so far: " + str(highestAccuracy) + 
                    ", Parameters: a=" + str(elementInC) +
                    ", degree=" + str(elementInDegree) +
                    ", kernel= " + str(elementInKernel) +
                    ", decision_function_shape = " + str(elementInDFS))
