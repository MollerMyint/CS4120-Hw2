#-------------------------------------------------------------------------
# AUTHOR: Moller Myint 
# FILENAME: knn.py
# SPECIFICATION: compute the LOO-CV error rate for a 1NN classifier on the spam/ham classification task
# FOR: CS 4210- Assignment #2
# TIME SPENT: how long it took you to complete the assignment: 3 hours
#-----------------------------------------------------------*/

#IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH AS numpy OR pandas. You have to work here only with standard vectors and arrays

#Importing some Python libraries
from sklearn.neighbors import KNeighborsClassifier
import csv

db = []

#Reading the data in a csv file
with open('email_classification.csv', 'r') as csvfile:
  reader = csv.reader(csvfile)
  for i, row in enumerate(reader):
      if i > 0: #skipping the header
         db.append(row)

error_count = 0

#Loop your data to allow each instance to be your test set
for i, row in enumerate(db):
    # Add the training features to the 20D array X removing the instance that will be used for testing in this iteration.
    X = []
    Y = []
    
    for j, data_point in enumerate(db):
        if j != i:
            X.append([float(feature) for feature in data_point[:-1]])  # All features except the last one
            # Transform the original training classes to numbers and add them to the vector Y.
            Y.append(1 if data_point[-1] == 'spam' else 0)  # Assuming last column is the label (spam/ham)
    
    # Store the test sample of this iteration in the vector testSample
    testSample = [float(feature) for feature in row[:-1]]  # All features except the last one
    testLabel = 1 if row[-1] == 'spam' else 0  # The label of the current test instance (spam/ham)

    # Fitting the knn to the data
    clf = KNeighborsClassifier(n_neighbors=1, p=2)
    clf = clf.fit(X, Y)

    # Use your test sample in this iteration to make the class prediction.
    class_predicted = clf.predict([testSample])[0]

    # Compare the prediction with the true label of the test instance to start calculating the error rate.
    if class_predicted != testLabel:
        error_count += 1

# Print the error rate
error_rate = error_count / len(db)
print(f"Error rate: {error_rate}")


