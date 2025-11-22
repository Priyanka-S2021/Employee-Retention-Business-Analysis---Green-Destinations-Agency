import pandas as pd
import numpy as np

df = pd.read_csv("/content/greendestination (1).csv")

df.head(10)

# Display basic information about the dataset
df.info()

# Display basic statistics
df.describe()

# Check for missing values
df.isnull().sum()

# Calculate overall attrition rate
attrition_rate = df['Attrition'].value_counts(normalize=True)['Yes']
print(f"Attrition Rate: {attrition_rate:.2%}")

import matplotlib.pyplot as plt
import seaborn as sns
# Explore the distribution of age
sns.histplot(df['Age'], kde=True)
plt.show()

# Explore the distribution of years at the company
sns.histplot(df['YearsAtCompany'], kde=True)
plt.show()

# Explore the distribution of income
sns.histplot(df['MonthlyIncome'], kde=True)
plt.show()

# Compare attrition rates across different categories
sns.countplot(x='Age', hue='Attrition', data=df, edgecolor='white')  # Plot with white edges
plt.locator_params(axis='x', nbins=10)
plt.show()

sns.countplot(x='YearsAtCompany', hue='Attrition', data=df, edgecolor='white')  # Plot with white edges
plt.locator_params(axis='x', nbins=10)
plt.show()

sns.countplot(x='MonthlyIncome', hue='Attrition', data=df)  # Plot with white edges
xtick_locations = [700]  # Replace this with your desired tick locations
plt.xticks(xtick_locations)
plt.show()



df.shape

import matplotlib.pyplot as plt
import seaborn as sns
# Increase figure size for better visibility
plt.figure(figsize=(16, 12))

# Correlation matrix (including all columns)
correlation_matrix = df.corr()

# Heatmap with rotated axis labels
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5, annot_kws={"size": 8}, xticklabels=correlation_matrix.columns, yticklabels=correlation_matrix.columns, cbar_kws={"shrink": 1})

plt.show()

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

# Assuming 'Attrition' is the target variable and you have categorical variables in X
# Use one-hot encoding for categorical variables
X_encoded = pd.get_dummies(X, drop_first=True)

# Split the data into training and testing sets
X_train_encoded, X_test_encoded, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

# Build a logistic regression model
model = LogisticRegression()
model.fit(X_train_encoded, y_train)

# Predictions
predictions = model.predict(X_test_encoded)

# Evaluate the model
print(confusion_matrix(y_test, predictions))
print(classification_report(y_test, predictions))
