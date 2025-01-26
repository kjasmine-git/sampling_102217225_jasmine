import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
import math
import requests

# Downloading the dataset
data_url = "https://github.com/AnjulaMehto/Sampling_Assignment/raw/main/Creditcard_data.csv"
file_name = "Creditcard_data.csv"
response = requests.get(data_url)
with open(file_name, "wb") as f:
    f.write(response.content)

# Loading the dataset
data = pd.read_csv(file_name)

# Separating features and target
X = data.drop("Class", axis=1)
y = data["Class"]

# Balancing the dataset
# Using oversampling
ros = RandomOverSampler(random_state=42)
X_oversampled, y_oversampled = ros.fit_resample(X, y)
X_balanced, y_balanced = X_oversampled, y_oversampled

# Sampling size formulas
Z = 1.96 
p = 0.5   
E = 0.05
S = 5
C = 10

n_simple = int((Z**2 * p * (1-p)) / E**2)
n_stratified = int((Z**2 * p * (1-p)) / (E/S)**2)  
n_cluster = int((Z**2 * p * (1-p)) / (E/C)**2)    
n_systematic = len(X_balanced) // 20

n_simple = min(n_simple, len(X_balanced))
n_stratified = min(n_stratified, len(X_balanced))
n_cluster = min(n_cluster, len(X_balanced))
n_systematic = min(n_systematic, len(X_balanced))

# Generating samples
simple_sample = X_balanced.sample(n_simple, random_state=42)
stratified_sample = X_balanced.sample(n_stratified, random_state=42)
cluster_sample = X_balanced.sample(n_cluster, random_state=42)
systematic_sample = X_balanced.iloc[::20]
random_sample = X_balanced.sample(100, random_state=42)  # Random sampling example

# Splitting data for ML models
samples = [simple_sample, stratified_sample, cluster_sample, systematic_sample, random_sample]
sample_names = ["Simple", "Stratified", "Cluster", "Systematic", "Random"]
models = [
    ("Random Forest", RandomForestClassifier(random_state=42)),
    ("Logistic Regression", LogisticRegression(max_iter=1000)),
    ("SVM", SVC()),
    ("Decision Tree", DecisionTreeClassifier(random_state=42)),
    ("KNN", KNeighborsClassifier()),
]

# Evaluate models
results = []
for sample_name, sample in zip(sample_names, samples):
    X_sample = sample
    y_sample = y_balanced.loc[sample.index]

    X_train, X_test, y_train, y_test = train_test_split(X_sample, y_sample, test_size=0.2, random_state=42)

    for model_name, model in models:
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        results.append({"Model": model_name, "Sampling Technique": sample_name, "Accuracy": accuracy})

# Create a DataFrame for results
df_results = pd.DataFrame(results)
pivot_table = df_results.pivot(index="Model", columns="Sampling Technique", values="Accuracy")

# Save results to CSV
pivot_table.to_csv("model_sampling_results_pivot.csv")
print(pivot_table)
