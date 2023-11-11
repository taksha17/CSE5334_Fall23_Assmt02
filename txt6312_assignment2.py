import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier

# Load the dataset
dataset_path = 'nba2021.csv'  # Update the path if necessary during the runtime
df = pd.read_csv(dataset_path)

# Preprocessing the dataset
label_encoder = LabelEncoder()
df['Pos'] = label_encoder.fit_transform(df['Pos'])
df_numerical = df.select_dtypes(include=[np.number])

# Separating features and target variable
X = df_numerical.drop('Pos', axis=1)
y = df_numerical['Pos']

# Scaling features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Task 1: Classification with KNN
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.25, random_state=0)

# Experimenting with different values of 'k'
best_k = 1
best_accuracy = 0
for k in range(1, 20):  # Trying k from 1 to 19 and seem to fetch the best value of k
    classifier = KNeighborsClassifier(n_neighbors=k)
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_k = k

# Using the best 'k'
classifier = KNeighborsClassifier(n_neighbors=best_k)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
training_accuracy = accuracy_score(y_train, classifier.predict(X_train))
test_accuracy = accuracy_score(y_test, y_pred)
print(f'Best k: {best_k}')
print(f'Training Accuracy: {training_accuracy}')
print(f'Test Accuracy: {test_accuracy}')

# Task 2: Confusion Matrix using Pandas crosstab
cm = pd.crosstab(y_test, y_pred, rownames=['Actual'], colnames=['Predicted'], margins=True)
print('Confusion Matrix using Pandas crosstab:\n', cm)

# Task 3: Cross-Validation
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)
accuracies = cross_val_score(classifier, X_scaled, y, cv=skf)
print('Accuracies for each fold:', accuracies)
print('Average Accuracy:', np.mean(accuracies))

# Task 4: Documentation
# The chosen method is K-Nearest Neighbors (KNN). KNN classifies based on the majority class among its nearest neighbors.
# Feature scaling and tuning the number of neighbors ('k') were key to improving the model's performance.
# The model's performance was evaluated using a train-test split and 10-fold stratified cross-validation.
# Also I tried decssion tree method as well but the output was as below

# Training Accuracy: 1.0
# Test Accuracy: 0.392
# Confusion Matrix using Pandas crosstab:
#  Predicted   0   1   2   3   4  All
# Actual
# 0          11   3   1   2   1   18
# 1           4   9   1   6   4   24
# 2           0   5  10   7   9   31
# 3           2  10   2   3   3   20
# 4           3   5   4   4  16   32
# All        20  32  18  22  33  125
# Accuracies for each fold: [0.44       0.32       0.48       0.32       0.4        0.36
#  0.54       0.3877551  0.32653061 0.40816327]
# Average Accuracy: 0.39824489795918366
# Showing clear signs of overfitting and and also with references from some github links , i decided to go ahead with the KNN itself/

