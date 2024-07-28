# Importing Libraries
import numpy as np
import matplotlib as plt
import pandas as pd

# Importing dataset
df = pd.read_csv('50_Startups.csv')
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

# Encoding the categorical data
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
state_idx = df.columns.get_loc('State')
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [state_idx])], remainder='passthrough')
X = np.array(ct.fit_transform(X))

# Splitting the Dataset
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Training the model
from sklearn.linear_model import LinearRegression
rg = LinearRegression()
rg.fit(X_train, y_train)