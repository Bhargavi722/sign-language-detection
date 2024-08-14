import tkinter as tk
from tkinter import filedialog
import cv2
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

# Load the trained model
model = torch.load(r"C:\Users\gowda\OneDrive\Desktop\sign_\model.ipynb")

# Create the GUI
root = tk.Tk()
root.title("Sign Language Prediction Model")

# Function to upload image
def upload_image():
    filepath = filedialog.askopenfilename()
    img = cv2.imread(filepath)
    # Preprocess the image
    img = cv2.resize(img, (224, 224))
    img = img / 255.0
    img = torch.tensor(img).permute(2, 0, 1)
    # Make prediction
    output = model(img.unsqueeze(0))
    _, predicted = torch.max(output, 1)
    result = predicted.item()
    # Display the result
    result_label.config(text=f"Predicted sign language: {result}")

# Function to upload video
def upload_video():
    filepath = filedialog.askopenfilename()
    cap = cv2.VideoCapture(filepath)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Preprocess the frame
        frame = cv2.resize(frame, (224, 224))
        frame = frame / 255.0
        frame = torch.tensor(frame).permute(2, 0, 1)
        # Make prediction
        output = model(frame.unsqueeze(0))
        _, predicted = torch.max(output, 1)
        result = predicted.item()
        # Display the result
        result_label.config(text=f"Predicted sign language: {result}")
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

# Create buttons to upload image and video
image_button = tk.Button(root, text="Upload Image", command=upload_image)
image_button.pack()

video_button = tk.Button(root, text="Upload Video", command=upload_video)
video_button.pack()

# Create label to display the result
result_label = tk.Label(root, text="")
result_label.pack()

root.mainloop()



