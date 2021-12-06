#!/usr/bin/env python
# coding: utf-8

# In[1]:


# The Train Test and Split Method Definition sklearn.model_selection.train_test_split(*arrays, test_size=None, train_size=None, random_state=None, shuffle=True, stratify=None)
# test_size: Value should be between 0.0 and 1.0 and signifies the percent of data we are putting under test data
# train_size: Like test size for training data divison

# Implementing it using a numpy array
import numpy as np
from sklearn.model_selection import train_test_split
X, y = np.arange(10).reshape((5, 2)), range(5)
X


# In[2]:


list(y)


# In[4]:


# Doing a train test and split on the data with a test size of 43% of input data
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.43)
# Check all the 4 individual arrays to see that the input has been split
X_train


# In[5]:


y_train


# In[6]:


X_test


# In[8]:


y_test
# This process helps in avoiding underfitting and overfitting problems with data


# In[9]:


# Implementing the Breast Cancer Data set [Code Snippet 2]
# Importing the Sci-kit learn library and importing Wisconsin breast cancer dataset
import sklearn
from sklearn.datasets import load_breast_cancer
data_set = load_breast_cancer()


# In[10]:


# Differentiating features and labels on the imported data set
label_names = data_set['target_names']
labels = data_set['target']
feature_names = data_set['feature_names']
features = data_set['data']

# We observe the that are two categories of cancer that need to be labelled (Output Parameters)
print(label_names)


# In[11]:


# Printing the raw data from the data-set and using .DESCR function to retrieve the descriptions of columns present in the data
print(data_set.data)


# In[12]:


print(data_set.DESCR)


# In[13]:


# Import pandas to use data frames for exploration
# Read the DataFrame, using the feature names from the data set
import pandas as pd
df = pd.DataFrame(data_set.data, columns=data_set.feature_names)
# Add a column to generate the ouput
df['target'] = data_set.target

# Use the head() method from Pandas to see the top of the data set
df.head()


# In[14]:


# Import the seaborn library to visualize the input data
# We will form a box plot to understand the spread of the data set in comparison to the target column
import seaborn as sns
box_plot = data_set.data
box_target = data_set.target
sns.boxplot(data = box_plot,width=0.6,fliersize=7)
sns.set(rc={'figure.figsize':(2,15)})


# In[15]:


# We will now move towards applying Machine Learning to this data
# Currently we know the target variables, and we have labelled input data as well.
# Use the train_test_split to create the training and testing data sets
from sklearn.model_selection import train_test_split
train, test, train_labels, test_labels = train_test_split(features,labels,test_size = 0.33, random_state = 40)

# Import the Naïve Bayes Classifier from Sci-kit learn’s library
from sklearn.naive_bayes import GaussianNB
g_nb = GaussianNB()

# The fit method is used to train the model using the training data
model = g_nb.fit(train, train_labels)
# The predict method runs the learned model on the test data to generate the output
predicts = g_nb.predict(test)
print(predicts)


# In[17]:


# We get an output that predicts the type of cancer ('malignant' or 'benign') caused from the given inputs

# Every time an algorithm is created and run, use the accuracy_score to check the working of that algorithm
from sklearn.metrics import accuracy_score
print(accuracy_score(test_labels,predicts))


# In[18]:


# K-Means Clustering Algorithm [Code Snippet 3]
# Let us now move towards creating a Clustering Algorithm in Python
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import numpy as np
# Import the KMeans package from Sci-Kit Learn
from sklearn.cluster import KMeans

# Creating randomly generated data in the form of Blobs
# Total Samples = 1000, Total Centers = 6, Standard Deviation of Clusters = 0.40
from sklearn.datasets.samples_generator import make_blobs
X, y_true = make_blobs(n_samples=1000, centers=6, cluster_std=0.40)

# Plot the clusters on a Seaborn Scatterplot
plt.scatter(X[:, 0], X[:, 1], s=25);
plt.show()


# In[19]:


# We see six distinct clusters on the plot
# Let us now import KMeans and find the centroids of these clusters
kmeans = KMeans(n_clusters=6)

# Fit the model with the generated data
kmeans.fit(X)
# Run the model on the given data plot the result graph
y_kmeans = kmeans.predict(X)
plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=75, cmap='viridis')
centers = kmeans.cluster_centers_


# In[20]:


# We can now see the clusters uniquely identifiable
# Checking implementation through centroids
plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5);
plt.show()


# In[21]:


# Import all required libraries along with Sci-kit learn metrics function 
import matplotlib.pyplot as plt
import seaborn as sns; 
import numpy as np
from sklearn.cluster import KMeans
# The metrics function will help us use the silhouette score construct
from sklearn import metrics

# Create the sample space using make_blobs method
from sklearn.datasets.samples_generator import make_blobs
X, y_true = make_blobs(n_samples=1500, centers=8, cluster_std=0.60)
# We have created a sample space with 1500 entries and 8 centers

# Create an empty numpy array and another array with evenly spaced values
scores = []
values = np.arange(4, 20)

# Iterate through the array and initialize each of the values for clustering with given k value
# 'k-means++' : selects initial cluster centers for k-mean clustering in a smart way to speed up convergence.
for num_clusters in values:
    kmeans = KMeans(init='k-means++', n_clusters=num_clusters, n_init=10)
    kmeans.fit(X)

# Import the silhouette score metric and use the Euclidean distance to check for ideal centroids
score = metrics.silhouette_score(X, kmeans.labels_, metric='euclidean', sample_size=len(X))

# The metric will now update the number of clusters and give the score
print("\nNumber of clusters =", num_clusters)
print("Silhouette score =", score)
scores.append(score)


# In[22]:


# Using numpy’s argmax function, we can now get the number of optimal clusters needed for this sample data 
num_clusters = np.argmax(scores) + values[0]
print('\nIdeal number of clusters =', num_clusters)


# In[ ]:




