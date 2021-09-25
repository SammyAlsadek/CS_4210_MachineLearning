#-------------------------------------------------------------------------
# AUTHOR: Sammy Alsadek
# FILENAME: knn.py
# SPECIFICATION: Program that outputs the LOO-CV error rate for 1NN
# FOR: CS 4210- Assignment #2
# TIME SPENT: 20 minutes
#-----------------------------------------------------------*/

#IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH AS numpy OR pandas. You have to work here only with standard vectors and arrays

#importing some Python libraries
from sklearn.neighbors import KNeighborsClassifier
import csv

db = []

#reading the data in a csv file
with open('binary_points.csv', 'r') as csvfile:
  reader = csv.reader(csvfile)
  for i, row in enumerate(reader):
      if i > 0: #skipping the header
         db.append (row)

dictionary = { '+' : 0, '-' : 1 }
wrongCount = 0

#loop your data to allow each instance to be your test set
for i, instance in enumerate(db):

    #add the training features to the 2D array X removing the instance that will be used for testing in this iteration. For instance, X = [[1, 3], [2, 1,], ...]]. Convert each feature value to
    # float to avoid warning messages
    #--> add your Python code here
    X = []
    for j in range(len(db)):
        if i != j:
            temp = []
            for k in range(len(db[j])-1):
                temp.append(float(db[j][k]))
            X.append(temp)

    #transform the original training classes to numbers and add to the vector Y removing the instance that will be used for testing in this iteration. For instance, Y = [1, 2, ,...]. Convert each
    #  feature value to float to avoid warning messages
    #--> add your Python code here
    Y = []
    for j in range(len(db)):
        if i != j:
            Y.append(dictionary[db[j][2]])

    #store the test sample of this iteration in the vector testSample
    #--> add your Python code here
    testSample = []
    for j in range(len(instance)-1):
        testSample.append(float(instance[j]))

    #fitting the knn to the data
    clf = KNeighborsClassifier(n_neighbors=1, p=2)
    clf = clf.fit(X, Y)

    #use your test sample in this iteration to make the class prediction. For instance:
    #class_predicted = clf.predict([[1, 2]])[0]
    #--> add your Python code here
    class_predicted = clf.predict([testSample])[0]

    #compare the prediction with the true label of the test instance to start calculating the error rate.
    #--> add your Python code here
    if class_predicted != dictionary[instance[len(instance)-1]]:
        wrongCount += 1

#print the error rate
#--> add your Python code here
errorRate = wrongCount / len(db)
print("Error Rate = " + str(errorRate))






