import cv2
from keras.models import load_model
import numpy as np
from tkinter import *
from PIL import Image, ImageDraw

# Load the CNN model
model = load_model('model.h5')

# Define classes and labels
letter_count = {0: 'CHECK', 1: '01_ka', 2: '02_kha', 3: '03_ga', 4: '04_gha', 5: '05_kna', 6: '06_cha',
                7: '07_chha', 8: '08_ja', 9: '09_jha', 10: '10_yna',
                11: '11_taa', 12: '12_thaa', 13: '13_daa', 14: '14_dhaa', 15: '15_adna', 16: '16_tabala', 17: '17_tha',
                18: '18_da', 19: '19_dha', 20: '20_na', 21: '21_pa', 22: '22_pha',
                23: '23_ba', 24: '24_bha', 25: '25_ma', 26: '26_yaw', 27: '27_ra', 28: '28_la', 29: '29_waw', 30: '30_motosaw',
                31: '31_petchiryakha', 32: '32_patalosaw', 33: '33_ha',
                34: '34_chhya', 35: '35_tra', 36: '36_gya', 37: 'CHECK'}

# Global variables
prev_x, prev_y = None, None
canvas_width, canvas_height = 280, 280
line_color = 'black'
image_size = (32, 32)

# Tkinter window setup
window = Tk()
window.title("Draw Character")

# Canvas setup
canvas = Canvas(window, width=canvas_width, height=canvas_height, bg='white')
canvas.pack()

# Image setup for drawing
image = Image.new('L', (canvas_width, canvas_height), 'white')
draw = ImageDraw.Draw(image)

# Function to start drawing
def start_draw(event):
    global prev_x, prev_y
    prev_x, prev_y = event.x, event.y

# Function to draw lines
def draw_line(event):
    global prev_x, prev_y
    x, y = event.x, event.y
    canvas.create_line(prev_x, prev_y, x, y, fill=line_color, width=5)
    draw.line([prev_x, prev_y, x, y], fill=line_color, width=5)
    prev_x, prev_y = x, y

# Function to erase the canvas and label
def erase():
    global draw
    canvas.delete("all")
    draw.rectangle([0, 0, canvas_width, canvas_height], fill='white')
    predicted_label.config(text="")

# Function to predict the drawn character
def predict_character():
    # Resize the image
    resized_image = image.resize(image_size)
    
    # Convert to numpy array and invert colors
    img_array = np.array(resized_image, dtype=np.float32)
    img_array = 255 - img_array  # Invert the image

    # Normalize to [0, 1]
    img_array = img_array / 255.0

    # Check the intermediate image array
    print("Preprocessed Image Array Shape:", img_array.shape)
    print("Preprocessed Image Array:\n", img_array)

    # Reshape for the model
    img_array = img_array.reshape(1, 32, 32, 1)
    
    # Predict using the loaded CNN model
    pred_probab = model.predict(img_array)[0]
    pred_class = np.argmax(pred_probab)

    # Debugging print statements
    print("Prediction Probabilities:\n", pred_probab)
    print("Predicted Class:", pred_class)

    # Display the predicted character label
    predicted_label.config(text=f"Predicted character: {letter_count[pred_class]}")

# Bind mouse events to canvas
canvas.bind("<Button-1>", start_draw)
canvas.bind("<B1-Motion>", draw_line)

# Button to predict the drawn character
predict_button = Button(window, text="Predict Character", command=predict_character)
predict_button.pack()

# Button to erase the canvas
erase_button = Button(window, text="Erase", command=erase)
erase_button.pack()

# Label to display predicted character
predicted_label = Label(window, text="Predicted character: ")
predicted_label.pack()

# Start the Tkinter main loop
window.mainloop()
