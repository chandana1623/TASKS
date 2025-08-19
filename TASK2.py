import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

# Load Titanic dataset
titanic = sns.load_dataset("titanic")

# Quick view
print(titanic.head())
print(titanic.info())
# General summary
print(titanic.describe(include="all"))

# Numerical features summary
print(titanic.describe())

# Count of categorical variables
print(titanic['class'].value_counts())
# Age distribution
plt.figure(figsize=(8,5))
sns.histplot(titanic['age'].dropna(), kde=True, bins=30)
plt.title("Age Distribution")
plt.show()

# Fare distribution with outliers
plt.figure(figsize=(8,5))
sns.boxplot(x=titanic['fare'])
plt.title("Fare Boxplot")
plt.show()
# Correlation matrix
plt.figure(figsize=(10,6))
sns.heatmap(titanic.corr(numeric_only=True), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

# Pairplot of selected features
sns.pairplot(titanic[['age', 'fare', 'survived']], hue="survived")
plt.show()
# Survival count
sns.countplot(x='survived', data=titanic)
plt.title("Survival Count")
plt.show()

# Survival by class
sns.countplot(x='class', hue='survived', data=titanic)
plt.title("Survival by Passenger Class")
plt.show()

# Survival by gender
sns.countplot(x='sex', hue='survived', data=titanic)
plt.title("Survival by Gender")
plt.show()
# Interactive scatter plot Age vs Fare
fig = px.scatter(titanic, x="age", y="fare", color="survived",
                 hover_data=["class", "sex"])
fig.show()

# Interactive bar chart for class survival
fig = px.histogram(titanic, x="class", color="survived", barmode="group")
fig.show()
