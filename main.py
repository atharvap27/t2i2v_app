import os
import torch
import logging
from flask import Flask, render_template, request, redirect, url_for, send_from_directory
from werkzeug.utils import secure_filename

from PIL import Image
from diffusers import StableDiffusionImg2ImgPipeline
from diffusers.utils import export_to_video

app = Flask(__name__)

# Configure the upload folder and allowed extensions
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Function to check if the file has an allowed extension
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Function to generate video using GPU
def generate_video_with_gpu(text_input, image_path):
    # Load the image
    p = Image.open(image_path)
    init_image = p.convert("RGB")

    # Initialize the AI model with CUDA support
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        "CompVis/stable-diffusion-v1-4",
        variant="fp16",
        torch_dtype=torch.float16,
    ).to("cuda")

    pipe.enable_attention_slicing()

    # Generate video using CUDA
    torch.manual_seed(767)
    gen_images = pipe(prompt=text_input, num_images_per_prompt=1, image=init_image, strength=0.8, num_inference_steps=80, guidance_scale=15)

    # Save the generated video
    video_path = os.path.join(app.config['UPLOAD_FOLDER'], 'generated.mp4')
    export_to_video(gen_images.frames[0], video_path, fps=7)

    return video_path

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Check if a text input and an image file were provided
        if 'text_input' not in request.form or 'image' not in request.files:
            return render_template('index.html', error='Please provide both text and an image.')

        text_input = request.form['text_input']
        image_file = request.files['image']

        # Check if the text input is empty
        if not text_input:
            return render_template('index.html', error='Text input cannot be empty.')

        # Check if an image file was provided and has an allowed extension
        if image_file.filename == '':
            return render_template('index.html', error='No selected image file.')

        if not allowed_file(image_file.filename):
            return render_template('index.html', error='Invalid file format. Allowed formats: jpg, jpeg, png.')

        # Securely save the image file
        filename = secure_filename(image_file.filename)
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        image_file.save(image_path)

        try:
            # Generate the video using GPU
            video_path = generate_video_with_gpu(text_input, image_path)
            
            # Return the video filename or link to the frontend
            return render_template('index.html', success='Video generated successfully. Download the video below.', video_path=video_path)
        except Exception as e:
            # Handle any exceptions during processing
            return render_template('index.html', error=f'Error processing the request: {str(e)}')

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
