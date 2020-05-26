import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from matplotlib.colors import ListedColormap



file = "student-mat.csv"

dataset = pd.read_csv(file)
X = dataset.iloc[:, [30, 31]].values
y = dataset.iloc[:, -1].values
   
print("Violin plot")
sns.violinplot(dataset['sex'], dataset['G3'])
sns.despine()
 
   
fig=plt.figure()
ax = fig.add_subplot(1,1,1)
ax.boxplot(dataset['G3'])
plt.show()

print(dataset.describe())

# Splitting the dataset into the Training set and Test set
X_train, X_validation, y_train, y_validation = train_test_split(X, y, test_size = 0.25, random_state = 17)

# Scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_validation = sc.transform(X_validation)

# Fitting Model to Train set
classifier = SVC(kernel = 'poly', degree=3, random_state = 0)
classifier.fit(X_train, y_train)

# Predict the Test set results
prediction = classifier.predict(X_validation)


print("Accuracy: ",100*accuracy_score(y_validation, prediction),"%")
print("=============================")
