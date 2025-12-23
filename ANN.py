import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Load data
df = pd.read_csv("F://Att.csv")

X = df[['Attendance', 'StudyHours', 'PreviousScore']]
y = df[['Marks']]

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

# Scale inputs
x_scaler = StandardScaler()
X_train = x_scaler.fit_transform(X_train)
X_test = x_scaler.transform(X_test)

# Scale output
y_scaler = StandardScaler()
y_train_scaled = y_scaler.fit_transform(y_train)
y_test_scaled = y_scaler.transform(y_test)

# Model
model = Sequential([
    Dense(8, activation='relu', input_shape=(3,)),
    Dense(4, activation='relu'),
    Dense(1)
])

model.compile(
    optimizer='adam',
    loss='mse',
    metrics=['mae', 'mse']
)

# Train
model.fit(X_train, y_train_scaled, epochs=300, verbose=0)

# Evaluate (scaled)
loss, mae, mse = model.evaluate(X_test, y_test_scaled, verbose=0)

# Predict
y_pred_scaled = model.predict(X_test)
y_pred = y_scaler.inverse_transform(y_pred_scaled)
# Metrics on original scale
mae_real = mean_absolute_error(y_test, y_pred)
mse_real = mean_squared_error(y_test, y_pred)
rmse_real = np.sqrt(mse_real)
r2 = r2_score(y_test, y_pred)
print("Actual Marks:", y_test.values.flatten())
print("Predicted Marks:", y_pred.flatten())
print("\nEvaluation Metrics:")
print("MAE  :", mae_real)
print("MSE  :", mse_real)
print("RMSE :", rmse_real)
print("R²   :", r2)
