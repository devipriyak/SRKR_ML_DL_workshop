import tkinter as tk
from tkinter import filedialog, messagebox
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

# Global variables
model = None
scaler = None
features = ["Attendance", "StudyHours", "PreviousScore"]

# -------------------------------
# Load CSV and Train Model
# -------------------------------
def load_csv_and_train():
    global model, scaler

    try:
        file_path = filedialog.askopenfilename(
            filetypes=[("CSV files", "*.csv")]
        )

        if not file_path:
            return

        df = pd.read_csv(file_path)

        # Check required columns
        required_cols = features + ["Marks"]
        if not all(col in df.columns for col in required_cols):
            messagebox.showerror(
                "Error",
                "CSV must contain columns:\nAttendance, StudyHours, PreviousScore, Marks"
            )
            return

        # Handle missing values
        df.fillna(df.mean(), inplace=True)

        X = df[features]
        y = df["Marks"]

        # Scaling
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Train model
        model = LinearRegression()
        model.fit(X_scaled, y)

        status_label.config(text="CSV Loaded & Model Trained Successfully", fg="green")

    except Exception as e:
        messagebox.showerror("Error", str(e))

# -------------------------------
# Prediction Function
# -------------------------------
def predict_marks():
    try:
        if model is None:
            messagebox.showwarning("Warning", "Please load CSV first")
            return

        attendance = float(attendance_entry.get())
        study_hours = float(hours_entry.get())
        prev_score = float(prev_entry.get())

        new_data = [[attendance, study_hours, prev_score]]
        new_data_scaled = scaler.transform(new_data)

        prediction = model.predict(new_data_scaled)

        result_label.config(
            text=f"Predicted Marks: {prediction[0]:.2f}",
            fg="blue"
        )

    except:
        messagebox.showerror("Error", "Please enter valid numeric inputs")

# -------------------------------
# Tkinter GUI
# -------------------------------
root = tk.Tk()
root.title("Multiple Linear Regression - CSV")
root.geometry("500x480")

tk.Label(
    root, text="Multiple Linear Regression (CSV Based)",
    font=("Arial", 16, "bold")
).pack(pady=10)

# Load CSV Button
tk.Button(
    root, text="Load CSV & Train Model",
    font=("Arial", 12),
    command=load_csv_and_train
).pack(pady=10)

status_label = tk.Label(root, text="", font=("Arial", 11))
status_label.pack(pady=5)

# Inputs
tk.Label(root, text="Attendance (%)", font=("Arial", 12)).pack()
attendance_entry = tk.Entry(root, font=("Arial", 12))
attendance_entry.pack(pady=5)

tk.Label(root, text="Study Hours / Day", font=("Arial", 12)).pack()
hours_entry = tk.Entry(root, font=("Arial", 12))
hours_entry.pack(pady=5)

tk.Label(root, text="Previous Exam Score", font=("Arial", 12)).pack()
prev_entry = tk.Entry(root, font=("Arial", 12))
prev_entry.pack(pady=5)

# Predict Button
tk.Button(
    root, text="Predict Marks",
    font=("Arial", 12),
    command=predict_marks
).pack(pady=15)

# Result
result_label = tk.Label(root, text="", font=("Arial", 14))
result_label.pack(pady=10)

root.mainloop()
