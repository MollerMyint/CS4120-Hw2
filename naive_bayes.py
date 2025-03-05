#-------------------------------------------------------------------------
# AUTHOR: Moller Myint 
# FILENAME: naive_bayes.py
# SPECIFICATION: Classify tennis play based on weather features using Naive Bayes classifier
# FOR: CS 4210- Assignment #2
# TIME SPENT: how long it took you to complete the assignment: 3 hours
#-----------------------------------------------------------*/

#IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH AS numpy OR pandas. You have to work here only with standard
# dictionaries, lists, and arrays

#Importing some Python libraries
from sklearn.naive_bayes import GaussianNB
import csv

#Reading the training data in a csv file
db = []
with open('weather_training.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    next(reader)  # Skip the header
    for row in reader:
        db.append(row)

#Transform the original training features to numbers and add them to the 4D array X.
#For instance Sunny = 1, Overcast = 2, Rain = 3, X = [[3, 1, 1, 2], [1, 3, 2, 2], ...]]
X = []
for row in db:
    outlook = 1 if row[1] == 'Sunny' else 2 if row[1] == 'Overcast' else 3
    temperature = 1 if row[2] == 'Cool' else 2 if row[2] == 'Mild' else 3
    humidity = 1 if row[3] == 'Normal' else 2 if row[3] == 'High' else 3
    wind = 1 if row[4] == 'Weak' else 2  # Assume Strong = 2
    X.append([outlook, temperature, humidity, wind])

#Transform the original training classes to numbers and add them to the vector Y.
#For instance Yes = 1, No = 2, so Y = [1, 1, 2, 2, ...]
Y = [1 if row[5] == 'Yes' else 2 for row in db]

#Fitting the naive bayes to the data
clf = GaussianNB(var_smoothing=1e-9)
clf.fit(X, Y)

#Reading the test data in a csv file
test_db = []
with open('weather_test.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    next(reader)  # Skip the header
    for row in reader:
        test_db.append(row)

#Printing the header of the solution
print("Day Outlook Temperature Humidity Wind PlayTennis Confidence")

#Use your test samples to make probabilistic predictions. For instance: clf.predict_proba([[3, 1, 2, 1]])[0]
for row in test_db:
    outlook = 1 if row[1] == 'Sunny' else 2 if row[1] == 'Overcast' else 3
    temperature = 1 if row[2] == 'Cool' else 2 if row[2] == 'Mild' else 3
    humidity = 1 if row[3] == 'Normal' else 2 if row[3] == 'High' else 3
    wind = 1 if row[4] == 'Weak' else 2  # Assume Strong = 2
    test_sample = [outlook, temperature, humidity, wind]
    
    # Get prediction probabilities
    proba = clf.predict_proba([test_sample])[0]
    confidence = max(proba)
    predicted_class = "Yes" if proba[0] > proba[1] else "No"
    
    if confidence >= 0.75:
        print(f"{row[0]} {row[1]} {row[2]} {row[3]} {row[4]} {predicted_class} {confidence:.2f}")

