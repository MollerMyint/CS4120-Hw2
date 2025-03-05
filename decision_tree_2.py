#-------------------------------------------------------------------------
# AUTHOR: Moller Myint
# FILENAME: decision_tree_2.py
# SPECIFICATION: train, test, and output the performance of the 3 models created by using each training set
# FOR: CS 4210- Assignment #2
# TIME SPENT: how long it took you to complete the assignment: 3 hours
#-----------------------------------------------------------*/

#IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH AS numpy OR pandas. You have to work here only with standard
# dictionaries, lists, and arrays

#Importing some Python libraries
from sklearn import tree
import csv

dataSets = ['contact_lens_training_1.csv', 'contact_lens_training_2.csv', 'contact_lens_training_3.csv']

for ds in dataSets:

    dbTraining = []
    X = []
    Y = []

    #Reading the training data in a csv file
    with open(ds, 'r') as csvfile:
         reader = csv.reader(csvfile)
         for i, row in enumerate(reader):
             if i > 0: #skipping the header
                dbTraining.append (row)

    #Transform the original categorical training features to numbers and add to the 4D array X.
    #For instance Young = 1, Prepresbyopic = 2, Presbyopic = 3, X = [[1, 1, 1, 1], [2, 2, 2, 2], ...]]
    for data in dbTraining:
        # Age: Young = 1, Prepresbyopic = 2, Presbyopic = 3
        age = 1 if data[0] == 'Young' else 2 if data[0] == 'Prepresbyopic' else 3
        
        # Spectacle Prescription: Myope = 1, Hypermetrope = 2
        spectacle_prescription = 1 if data[1] == 'Myope' else 2
        
        # Astigmatism: Yes = 1, No = 2
        astigmatism = 1 if data[2] == 'Yes' else 2
        
        # Tear Production Rate: Normal = 1, Reduced = 2
        tear_production_rate = 1 if data[3] == 'Normal' else 2
        
        X.append([age, spectacle_prescription, astigmatism, tear_production_rate])
        
    #Transform the original categorical training classes to numbers and add to the vector Y.
    #For instance Yes = 1 and No = 2, Y = [1, 1, 2, 2, ...]
        recommended_lenses = 1 if data[4] == 'Yes' else 2
        Y.append(recommended_lenses)

    # Loop your training and test tasks 10 times here
    accuracy_list = []
    for i in range(10):

       # Fitting the decision tree to the data setting max_depth=5
       clf = tree.DecisionTreeClassifier(criterion='entropy', max_depth=5)
       clf = clf.fit(X, Y)

       # Simulate reading the test data (since we don't have the test file)
       dbTest = dbTraining

       dbTest = []
       with open('contact_lens_test.csv', 'r') as testfile:  # Change file name as needed
        test_reader = csv.reader(testfile)
        for j, row in enumerate(test_reader):
          if j > 0:  # Skip header
            dbTest.append(row)

       correct_predictions = 0
       for data in dbTest:
           # Transform the features of the test instances to numbers following the same strategy done during training
           age = 1 if data[0] == 'Young' else 2 if data[0] == 'Prepresbyopic' else 3
           spectacle_prescription = 1 if data[1] == 'Myope' else 2
           astigmatism = 1 if data[2] == 'Yes' else 2
           tear_production_rate = 1 if data[3] == 'Normal' else 2
           
           # Use the decision tree to make the class prediction
           class_predicted = clf.predict([[age, spectacle_prescription, astigmatism, tear_production_rate]])[0]

           #Compare the prediction with the true label (located at data[4]) of the test instance to start calculating the accuracy.
           true_label = 1 if data[4] == 'Yes' else 2
           if class_predicted == true_label:
               correct_predictions += 1

       # Calculate accuracy for this run
       accuracy = correct_predictions / len(dbTest)
       accuracy_list.append(accuracy)

    #Find the average of this model during the 10 runs (training and test set)
    avg_accuracy = sum(accuracy_list) / len(accuracy_list)


    #Print the average accuracy of this model during the 10 runs (training and test set).
    #Your output should be something like that: final accuracy when training on contact_lens_training_1.csv: 0.2
    print(f'final accuracy when training on {ds}: {avg_accuracy:.2f}')
