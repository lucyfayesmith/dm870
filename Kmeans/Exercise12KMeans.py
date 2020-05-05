import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn import metrics


df = pd.read_csv('R15.txt')
x = df.iloc[:,[0,1]].values

#Elbow
# Error = []
# for i in range(1,20):
#     kmeans = KMeans(n_clusters=i).fit(x)
#     kmeans.fit(x)
#     Error.append(kmeans.inertia_)

# plt.plot(range(1,20),Error)
# plt.title('Elbow method')
# plt.xlabel('No of clusters')
# plt.ylabel('Error')
# plt.show()


kmeans = KMeans(n_clusters=7)
y_kmeans = kmeans.fit_predict(x)

plt.scatter(x[:,0],x[:,1],c=y_kmeans)
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='black', marker="x");
plt.show()

labels = kmeans.labels_

print("Silhouette Coefficient: %0.3f"
      % metrics.silhouette_score(x, labels))
