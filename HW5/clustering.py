#-------------------------------------------------------------------------
# AUTHOR: Sammy Alsadek
# FILENAME: clustering.py
# SPECIFICATION: 
#    Completethe  Python  program  (clustering.py)  that  will  read  the  file training_data.csv  to cluster  the  data.  
#    Your  goal  is  to  run k-means  multiple  times  and  check  which  k  value  maximizes  the Silhouettecoefficient.  
#    You  also  need  to  plot  the  values  of  k  and  their  corresponding  Silhouettecoefficients so that we can visualize 
#    and confirm the best k value found. Next, you will calculateand print the Homogeneity score (the formulaof this evaluation 
#    metricis provided in the template) of this best  k clustering  task  by  using  the testing_data.csv,  which  is  a  file  
#    that  includes  ground  truth  data (classes).  Finally,  you  will  use  the  same  k  value  found  before  with  k-means  
#    to  run  Agglomerative clustering a single time, checking and printing its Homogeneity score as well
# FOR: CS 4210- Assignment #5
# TIME SPENT: 45 minutes
#-----------------------------------------------------------*/

#importing some Python libraries
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from sklearn import metrics

df = pd.read_csv('training_data.csv', sep=',', header=None) #reading the data by using Pandas library

#assign your training data to X_training feature matrix
X_training = df

#run kmeans testing different k values from 2 until 20 clusters
     #Use:  kmeans = KMeans(n_clusters=k, random_state=0)
     #      kmeans.fit(X_training)
     #--> add your Python code
silhouette_coefficients = []
for k in range(2, 21):
     kmeans = KMeans(n_clusters=k, random_state=0)
     kmeans.fit(X_training)

     #for each k, calculate the silhouette_coefficient by using: silhouette_score(X_training, kmeans.labels_)
     #find which k maximizes the silhouette_coefficient
     #--> add your Python code here
     silhouette_coefficient = silhouette_score(X_training, kmeans.labels_)
     silhouette_coefficients.append(silhouette_coefficient)

#plot the value of the silhouette_coefficient for each k value of kmeans so that we can see the best k
#--> add your Python code here
k_value = [k for k in range(2, 21)]
plt.plot(k_value, silhouette_coefficients)
plt.show()

#reading the validation data (clusters) by using Pandas library
#--> add your Python code here
testing_data = pd.read_csv('testing_data.csv', header=None)

#assign your data labels to vector labels (you might need to reshape the row vector to a column vector)
# do this: np.array(df.values).reshape(1,<number of samples>)[0]
#--> add your Python code here
labels = np.array(testing_data.values).reshape(1, -1)[0]

#Calculate and print the Homogeneity of this kmeans clustering
print("K-Means Homogeneity Score = " + metrics.homogeneity_score(labels, kmeans.labels_).__str__())
#--> add your Python code here
max_silhouette_coefficient = (max(silhouette_coefficients))
best_k_value = k_value[silhouette_coefficients.index(max_silhouette_coefficient)]

#rung agglomerative clustering now by using the best value o k calculated before by kmeans
#Do it:
agg = AgglomerativeClustering(n_clusters=best_k_value, linkage='ward')
agg.fit(X_training)

#Calculate and print the Homogeneity of this agglomerative clustering
print("Agglomerative Clustering Homogeneity Score = " + metrics.homogeneity_score(labels, agg.labels_).__str__())