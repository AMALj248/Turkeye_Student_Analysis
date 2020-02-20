import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv('student.csv')

#getting information about datset

print(data.head(10))
print(data.describe())
data.info()
#checking for null values
print(data.isnull())
#since no null values we can proceed without cleaning the data

#now plotting the dataset for better understanding
plt.figure(figsize=(20, 6))
sns.countplot(x='class', data=data)
plt.show()




#lets begin cluster the students on the basis of questionaire
data_questions=data.iloc[:,5:33]


#lets do a PCA for feature dimensional reduction
from sklearn.decomposition import PCA


pca = PCA(n_components = 2)
dataset_questions_pca = pca.fit_transform(data_questions)
# measurement is Within Cluster Sum of Squares (WCSS)
from sklearn.cluster import KMeans
wcss = []
for i in range(1, 7):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
    kmeans.fit(dataset_questions_pca)
    wcss.append(kmeans.inertia_)
#he KMeans algorithm clusters data by trying to separate samples in n groups of equal variance, minimizing a criterion known as the inerti
plt.plot(range(1, 7), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

kmeans=KMeans(n_clusters=3,init='k-means++')
y_kmeans=kmeans.fit_predict(dataset_questions_pca)


plt.scatter(dataset_questions_pca[y_kmeans==0,0],dataset_questions_pca[y_kmeans==0,1],s=100,c='yellow',label='cluster1')
plt.scatter(dataset_questions_pca[y_kmeans==1,0],dataset_questions_pca[y_kmeans==1,1],s=100,c='green',label='cluster2')
plt.scatter(dataset_questions_pca[y_kmeans==2,0],dataset_questions_pca[y_kmeans==2,1],s=100,c='red',label='cluster3')
plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],s=300,c='blue',label='Centroids')
plt.title('cluster of students')
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.legend()
plt.show()



