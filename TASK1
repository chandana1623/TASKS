import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset (Titanic dataset example from seaborn)
titanic = sns.load_dataset("titanic")

# Display first rows
print(titanic.head())
print(titanic.info())
# Check null values
print(titanic.isnull().sum())

# Fill numerical missing values (Age) with median
titanic['age'].fillna(titanic['age'].median(), inplace=True)

# Fill categorical missing values (Embarked) with mode
titanic['embarked'].fillna(titanic['embarked'].mode()[0], inplace=True)

# Drop irrelevant column with too many missing values (deck)
titanic.drop(columns=['deck'], inplace=True)
# Convert 'sex' and 'embarked' into numerical using one-hot encoding
titanic = pd.get_dummies(titanic, columns=['sex', 'embarked'], drop_first=True)

print(titanic.head())
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

# Choose numerical columns to scale
num_cols = ['age', 'fare']

titanic[num_cols] = scaler.fit_transform(titanic[num_cols])

print(titanic[num_cols].head())
# Boxplot to detect outliers
plt.figure(figsize=(10,5))
sns.boxplot(x=titanic['fare'])
plt.show()

# Remove outliers using IQR
Q1 = titanic['fare'].quantile(0.25)
Q3 = titanic['fare'].quantile(0.75)
IQR = Q3 - Q1

titanic = titanic[(titanic['fare'] >= Q1 - 1.5*IQR) & (titanic['fare'] <= Q3 + 1.5*IQR)]
print(titanic.head())
print(titanic.info())
