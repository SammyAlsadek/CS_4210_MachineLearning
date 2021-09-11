# -------------------------------------------------------------------------
# AUTHOR:           Sammy Alsadek
# FILENAME:         find_s.py
# SPECIFICATION:    Program that will read the file contact_lens.csv 
#                   and output the hypothesis of Find-S algorithm
# FOR:              CS 4200 - Assignment #1
# TIME SPENT:       Around 20 minutes
# -----------------------------------------------------------*/

# IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH AS numpy OR pandas. You have to work here only with standard vectors and arrays

# importing some Python libraries
import csv

num_attributes = 4
db = []
print("\n The Given Training Data Set \n")

# reading the data in a csv file
with open('contact_lens.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    for i, row in enumerate(reader):
        if i > 0:  # skipping the header
            db.append(row)
            print(row)

print("\n The initial value of hypothesis: ")
# representing the most specific possible hypothesis
hypothesis = ['0'] * num_attributes
print(hypothesis)

# find the first positive training data in db and assign it to the vector hypothesis
# --> add your Python code here
for row in db:
    if row[4] == "Yes":
        for i in range(0,len(hypothesis)):
            hypothesis[i] = row[i]
        break

# find the maximally specific hypothesis according to your training data in db and assign it to the vector hypothesis (special characters allowed: "0" and "?")
# --> add your Python code here
for row in db:
    if row[4] == "Yes":
        for i in range(0,len(hypothesis)):
            if hypothesis[i] != "?" and hypothesis[i] != row[i]:
                hypothesis[i] = "?"

print("\n The Maximally Specific Hypothesis for the given training examples found by Find-S algorithm:\n")
print(hypothesis)
