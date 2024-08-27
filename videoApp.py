import tkinter as tk
import customtkinter as ctk
from PIL import Image, ImageTk, ImageDraw, ImageFont
from authtoken import auth_token
import torch
from torch import autocast
from diffusers import StableDiffusionPipeline
import os
import cv2  # Install opencv-python
import numpy as np

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
counter = 1

def generate():
    global counter

    # Clear previous images
    if os.path.exists("images"):
        for f in os.listdir("images"):
            os.remove(os.path.join("images", f))
    else:
        os.makedirs("images")

    # Parse the text into sentences
    text = prompt.get()
    sentences = [s.strip() for s in text.split('.') if s.strip()]

    image_files = []

    for i, sentence in enumerate(sentences):
        with autocast(device):
            # Generate the image with the sentence
            output = pipe(sentence, guidance_scale=8.5)

        image = output.images[0]
        draw = ImageDraw.Draw(image)

        # Specify the path to your TTF font file and the desired font size
        font_path = "C:/Windows/Fonts/arial.ttf"
        font_size = 40
        try:
            font = ImageFont.truetype(font_path, font_size)
        except OSError:
            print(f"Cannot open font resource at {font_path}")
            return

        # Get text size and calculate position
        text_bbox = draw.textbbox((0, 0), sentence, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        image_width, image_height = image.size
        position = ((image_width - text_width) / 2, (image_height - text_height) / 2)

        # Draw text on the image
        draw.text(position, sentence, font=font, fill="white")

        # Save the image
        filename = f'images/generatedimage_{counter}.png'
        image.save(filename)
        image_files.append(filename)
        counter += 1

    # Combine images into a video
    combine_images_to_video(image_files)

    # Display the first image
    img = ImageTk.PhotoImage(Image.open(image_files[0]))
    lmain.configure(image=img)
    lmain.image = img

# display_duration a frame = 3s
def combine_images_to_video(image_files, output_video='output_video.mp4', display_duration=3):
    if not image_files:
        print("No images to combine into a video.")
        return

    # Read the first image to get the frame size
    frame = cv2.imread(image_files[0])
    height, width, layers = frame.shape

    # Set the frame rate (FPS) to display each frame for `display_duration` seconds
    fps = 1 / display_duration
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

    for image_file in image_files:
        frame = cv2.imread(image_file)
        video.write(frame)

    video.release()
    print(f"Video saved as {output_video}")

trigger = ctk.CTkButton(master=app, height=40, width=120, font=("Arial", 20), text_color="white", fg_color="blue",
                        command=generate)
trigger.configure(text="Generate")
trigger.place(x=206, y=60)

app.mainloop()
