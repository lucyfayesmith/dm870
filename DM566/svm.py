import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn import svm



#import the data

adult_data = pd.read_csv("adult.csv")


#Analysis and Visualisation of dataset

X = adult_data.iloc[:, [0, 4]].values
y = adult_data.iloc[:, -1].values

# Taking different features to acheive an optimal score which reflects the accuracy
X = adult_data[['age', 'education.num']]

# Taking the labels (Income)
Y = adult_data['income']

# Splitting into 80% for training set and 20% for testing set so we can see our accuracy, therfore 0.8 is optimal score
X_train, x_test, Y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

# Declaring the SVC with no tuning
classifier = SVC()

# Fitting the data. This is where the SVM will learn
classifier.fit(X_train, Y_train)


# Generate scatter plot for training data
plt.scatter(X_train.iloc[:,0], X_train.iloc[:,1])
plt.title('Linearly separable data')
plt.xlabel('Age')
plt.ylabel('Education.nu ')
plt.show()

# Initialize SVM classifier
clf = svm.SVC(kernel='poly')

# Fit data
clf = clf.fit(X_train, Y_train)

# Get support vector indices
support_vector_indices = clf.support_

# Get support vectors themselves
support_vectors = clf.support_vectors_

# Visualize support vectors
plt.scatter(X_train.iloc[:,0], X_train.iloc[:,1])
plt.scatter(support_vectors[:,0], support_vectors[:,1], color='red')
plt.title('Linearly separable data with support vectors')
plt.xlabel('Age')
plt.ylabel('Education.num')
plt.show()