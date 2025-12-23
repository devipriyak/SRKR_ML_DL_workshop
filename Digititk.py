import tkinter as tk
from PIL import Image, ImageDraw, ImageOps
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

# --- 1. Train a small CNN (for demo) ---
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(-1,28,28,1)/255.0
y_train_cat = to_categorical(y_train, 10)

model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
    MaxPooling2D((2,2)),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D((2,2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train_cat, epochs=2, batch_size=128, verbose=0)

# --- 2. Tkinter GUI ---
class DigitApp:
    def __init__(self, master):
        self.master = master
        master.title("CNN Digit Recognizer")

        self.canvas = tk.Canvas(master, width=200, height=200, bg='white')
        self.canvas.pack()

        self.button_predict = tk.Button(master, text="Predict", command=self.predict_digit)
        self.button_predict.pack()

        self.button_clear = tk.Button(master, text="Clear", command=self.clear_canvas)
        self.button_clear.pack()

        self.label = tk.Label(master, text="")
        self.label.pack()

        self.image1 = Image.new("L", (200,200), color=255)
        self.draw = ImageDraw.Draw(self.image1)
        self.canvas.bind("<B1-Motion>", self.paint)

    def paint(self, event):
        x1, y1 = (event.x-8), (event.y-8)
        x2, y2 = (event.x+8), (event.y+8)
        self.canvas.create_oval(x1, y1, x2, y2, fill='black')
        self.draw.ellipse([x1, y1, x2, y2], fill=0)

    def clear_canvas(self):
        self.canvas.delete("all")
        self.draw.rectangle([0,0,200,200], fill=255)
        self.label.config(text="")

    def predict_digit(self):
        img = self.image1.resize((28,28))
        img = ImageOps.invert(img)
        img = np.array(img)/255.0
        img = img.reshape(1,28,28,1)
        pred = model.predict(img)
        digit = np.argmax(pred)
        self.label.config(text=f"Predicted Digit: {digit}")

root = tk.Tk()
app = DigitApp(root)
root.mainloop()
