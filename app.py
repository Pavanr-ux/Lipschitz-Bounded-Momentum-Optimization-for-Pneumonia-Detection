import gradio as gr
import torch
from model import PneumoniaCNN
import numpy as np
from PIL import Image

# Load model
model = PneumoniaCNN()
model.load_state_dict(torch.load("model_lipschitz.pth", map_location="cpu"))
model.eval()


def predict(image):

    if image is None:
        return "Upload image first"

    try:
        # image from gradio is numpy

        # convert to grayscale
        if len(image.shape) == 3:
            image = image.mean(axis=2)

        # resize using PIL
        image = Image.fromarray(image.astype("uint8")).resize((224, 224))

        # normalize
        image = np.array(image) / 255.0

        # convert to tensor
        image = torch.tensor(image).float()

        # add channel dimension
        image = image.unsqueeze(0)

        # add batch dimension
        image = image.unsqueeze(0)

        with torch.no_grad():
            output = model(image)
            _, pred = torch.max(output, 1)

        if pred.item() == 1:
            return "🦠 Pneumonia Detected"
        else:
            return "✅ Normal"

    except Exception as e:
        print("ERROR:", e)
        return str(e)


gr.Interface(
    fn=predict,
    inputs=gr.Image(type="numpy"),
    outputs="text",
    title="Pneumonia Detection from X-ray",
    description="Upload a chest X-ray image to check for pneumonia."
).launch()
