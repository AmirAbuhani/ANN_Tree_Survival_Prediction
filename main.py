# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 13:16:17 2024

@author: AMIR
             
"""
# Importing necessary libraries
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

######################### Part 1 - Data Preprocessing #########################

# Import the dataset
dataset = pd.read_csv("Tree_Data.csv")

# Independent features (all columns except the first and the 3 last columns )
x = dataset.iloc[:, 1:-3].values
x = np.delete(x, 12, axis=1)  # Exclude the PlantDate column(index 12)
# target variable vector (Event Column)
y = dataset.iloc[:, -3].values

# Identify rows in y that are NA and drop these rows from both x and y
na_indices = np.where(pd.isna(y))[0]
x = np.delete(x, na_indices, axis=0)
y = np.delete(y, na_indices, axis=0)

# Cleaning data
# Convert non-numeric values to NaN in column 7 (Adult)
x[:, 7] = pd.to_numeric(x[:, 7], errors='coerce')

# Impute missing values
mean_imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
x[:, 7] = mean_imputer.fit_transform(x[:, 7].reshape(-1, 1)).flatten()
x[:, 13] = mean_imputer.fit_transform(x[:, 13].reshape(-1, 1)).flatten()


# One Hot Encoding for categorical features
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1, 2, 4, 6, 9, 11])], remainder='passthrough')
x = np.array(ct.fit_transform(x))

# Encoding specific categorical columns
le = LabelEncoder()
x[:, 27] = le.fit_transform(x[:, 27])  # Core column
x[:, 29] = le.fit_transform(x[:, 29])  # Sterile column
x[:, 30] = le.fit_transform(x[:, 30])  # Myco column


# Splitting the dataset into Training and Test sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)

# Feature Scaling
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

############################# Part 2 - Building the ANN #######################

# Initializing the ANN
ann = tf.keras.models.Sequential()

# Adding the input layer and the first hidden layer
ann.add(tf.keras.layers.Dense(units=6, activation='relu'))

# Adding the second hidden layer
ann.add(tf.keras.layers.Dense(units=6, activation='relu'))

# Adding the output layer
ann.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

######################### Part 3 - Training the ANN ###########################

# Compiling the ANN
ann.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Training the ANN on the Training set
ann.fit(x_train, y_train, batch_size=64, epochs=30)


######################## Part 4 - Saving the Model ###########################

# Save the trained model
ann.save('trained_model.h5')

######################### Part 5 - Making Predictions #########################
'''
To load the model(see the last part in the READMD.me file to more details):
model = tf.keras.models.load_model('trained_model.h5')
'''

# Making predictions on the Test set
y_pred = ann.predict(x_test)
# Convert probabilities to binary output
y_pred = (y_pred > 0.5)
