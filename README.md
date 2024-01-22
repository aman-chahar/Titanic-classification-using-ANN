# Titanic-classification-using-ANN


```python
# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import accuracy_score

# Load the Titanic dataset
train_data = pd.read_csv("/content/drive/MyDrive/Colab Notebooks/titanic/train.csv")
train_data.head()
```

Here, you are importing the required libraries and loading the Titanic dataset using Pandas.

```python
# One-hot encode the 'Embarked' column
ports = pd.get_dummies(train_data.Embarked, prefix='Embarked')
train_data = train_data.join(ports)
train_data.drop(['Embarked'], axis=1, inplace=True)
```

This part performs one-hot encoding on the 'Embarked' column, creating dummy variables for each port. It then joins these dummy variables with the original dataset and drops the original 'Embarked' column.

```python
# Map 'Sex' column to numerical values
train_data.Sex = train_data.Sex.map({'male': 0, 'female': 1})
```

Here, you are mapping the 'Sex' column to numerical values, where 'male' is mapped to 0 and 'female' is mapped to 1.

```python
# Prepare features (x) and target variable (y)
y = train_data.Survived.copy()
x = train_data.drop(['Survived'], axis=1)
x.drop(['Cabin', 'Ticket', 'Name', 'PassengerId'], axis=1, inplace=True)
```

This part separates the features (x) and the target variable (y) from the dataset. It also drops unnecessary columns like 'Cabin', 'Ticket', 'Name', and 'PassengerId'.

```python
# Handle missing values in 'Age' column
x.Age.fillna(x.Age.mean(), inplace=True)
```

Here, missing values in the 'Age' column are filled with the mean age.

```python
# Split the dataset into training and testing sets
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2)
```

The dataset is split into training and testing sets using `train_test_split` from scikit-learn.

```python
# Build a neural network model using TensorFlow and Keras
model = Sequential()
model.add(Dense(120, activation="relu", input_shape=(9,)))
# ... (multiple hidden layers)
model.add(Dense(1, activation="sigmoid"))
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
```

Here, a neural network model is defined using the Sequential API of Keras. It consists of an input layer, multiple hidden layers with ReLU activation, and an output layer with a sigmoid activation function. The model is compiled with the Adam optimizer and binary crossentropy loss.

```python
# Train the model on the training set
model.fit(xtrain, ytrain, batch_size=50, epochs=60)
```

The model is trained on the training set for 60 epochs with a batch size of 50.

```python
# Make predictions on the test set
ypred = model.predict(xtest)
ypred = (ypred >= 0.5).astype("int")
```

The model is used to make predictions on the test set, and the predictions are converted to binary values (0 or 1) based on a threshold of 0.5.

```python
# Evaluate the model accuracy
accuracy_score(ypred, ytest)
```

The accuracy of the model is evaluated using scikit-learn's `accuracy_score` function. The final accuracy score is 82%.

This code essentially builds a simple neural network using TensorFlow/Keras to predict survival on the Titanic dataset. The model is trained and evaluated on a subset of the data.
