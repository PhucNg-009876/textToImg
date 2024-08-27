import tkinter as tk
import customtkinter as ctk

from PIL import Image, ImageTk, ImageDraw, ImageFont
from authtoken import auth_token

import torch
from torch import autocast
from diffusers import StableDiffusionPipeline
import os
os.environ["HF_HOME"] = "D:/huggingface_cache"

# Create the app
app = tk.Tk()
app.geometry("532x632")
app.title("Stable Bud")
ctk.set_appearance_mode("dark")

prompt = ctk.CTkEntry(master=app, height=40, width=512, font=("Arial", 20), text_color="black", fg_color="white")
prompt.place(x=10, y=10)

lmain = ctk.CTkLabel(master=app, height=512, width=512)
lmain.place(x=10, y=110)

modelid = "stabilityai/stable-diffusion-2-1"
device = "cuda"
pipe = StableDiffusionPipeline.from_pretrained(modelid, variant="fp16", torch_dtype=torch.float16,
                                               use_auth_token=auth_token)
pipe.to(device)

# Initialize a counter
counter = 2


def generate():
    global counter  # Access the global counter variable

    with autocast(device):
        # Generate the image with the correct output structure
        output = pipe(prompt.get(), guidance_scale=8.5)

    # Access the image from the correct key
    image = output.images[0]

    # Create a drawing context
    draw = ImageDraw.Draw(image)

    # Specify the path to your TTF font file and the desired font size
    font_path = "C:/Windows/Fonts/arial.ttf"  # Replace with your TTF font path
    font_size = 40  # Adjust font size as needed
    font = ImageFont.truetype(font_path, font_size)

    # Text to overlay
    text = prompt.get()

    # Get text size and calculate position
    text_bbox = draw.textbbox((0, 0), text, font=font)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]
    image_width, image_height = image.size
    position = ((image_width - text_width) / 2, (image_height - text_height) / 2)

    # Draw text on the image
    draw.text(position, text, font=font, fill="white")

    # Create a unique filename using the counter
    filename = f'generatedimage_{counter}.png'

    # Save the image
    image.save(filename)

    # Increment the counter
    counter += 1

    # Display the image
    img = ImageTk.PhotoImage(image)
    lmain.configure(image=img)
    lmain.image = img  # Keep a reference to the image to prevent garbage collection


trigger = ctk.CTkButton(master=app, height=40, width=120, font=("Arial", 20), text_color="white", fg_color="blue",
                        command=generate)
trigger.configure(text="Generate")
trigger.place(x=206, y=60)

app.mainloop()
