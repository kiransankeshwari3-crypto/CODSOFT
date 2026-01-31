# Movie Rating Prediction - Fixed Version (For Your Dataset)

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# =========================
# 1. Load Dataset
# =========================
data = pd.read_csv("movies.csv")

print("Dataset Loaded Successfully!")
print("Shape of dataset:", data.shape)
print("\nColumns:")
print(data.columns)

# =========================
# 2. Select Correct Columns
# =========================
required_columns = ['Genre', 'Director', 'Actor 1', 'Rating']
data = data[required_columns]

# =========================
# 3. Handle Missing Values
# =========================
data = data.dropna()

# =========================
# 4. Encode Text Columns
# =========================
encoder = LabelEncoder()

data['Genre'] = encoder.fit_transform(data['Genre'])
data['Director'] = encoder.fit_transform(data['Director'])
data['Actor 1'] = encoder.fit_transform(data['Actor 1'])

# =========================
# 5. Split Features & Target
# =========================
X = data[['Genre', 'Director', 'Actor 1']]
y = data['Rating']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =========================
# 6. Train Model
# =========================
model = LinearRegression()
model.fit(X_train, y_train)

print("\nModel Training Completed!")

# =========================
# 7. Evaluate Model
# =========================
predictions = model.predict(X_test)

mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print("\nModel Evaluation:")
print("Mean Squared Error:", round(mse, 3))
print("R2 Score:", round(r2, 3))

# =========================
# 8. Predict Sample Movie Rating
# =========================
sample_movie = np.array([[1, 10, 5]])  # encoded Genre, Director, Actor
sample_prediction = model.predict(sample_movie)

print("\nSample Movie Rating Prediction:")
print("Predicted Rating:", round(sample_prediction[0], 2))
