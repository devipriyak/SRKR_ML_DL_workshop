import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
# Load data
df = pd.read_csv("F://Att.csv")

X = df[['Attendance', 'StudyHours', 'PreviousScore']]
y = df[['Marks']]   # keep as 2D for scaler

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

# Scale FEATURES
x_scaler = StandardScaler()
X_train = x_scaler.fit_transform(X_train)
X_test = x_scaler.transform(X_test)

# Scale TARGET (IMPORTANT)
y_scaler = StandardScaler()
y_train = y_scaler.fit_transform(y_train)
y_test = y_scaler.transform(y_test)
# Neural Network (Regression)
model = Sequential([
    Dense(8, activation='relu', input_shape=(3,)),
    Dense(4, activation='relu'),
    Dense(1)
])
model.compile(optimizer='adam', loss='mse', metrics=['mae'])
# Train
model.fit(X_train, y_train, epochs=100, verbose=0)
# Predict (SCALED)
y_pred_scaled = model.predict(X_test)
# INVERSE TRANSFORM predictions
y_pred = y_scaler.inverse_transform(y_pred_scaled)
y_test_actual = y_scaler.inverse_transform(y_test)
print("Actual Marks:", y_test_actual.flatten())
print("Predicted Marks:", y_pred.flatten())
