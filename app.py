import gradio as gr
import torch
from model import PneumoniaCNN
from torchvision import transforms

model = PneumoniaCNN()
model.load_state_dict(torch.load("model_lipschitz.pth", map_location="cpu"))
model.eval()

transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

def predict(image):
    image = transform(image).unsqueeze(0)
    output = model(image)

    _, pred = torch.max(output, 1)
    return "PNEUMONIA" if pred.item() == 1 else "NORMAL"

gr.Interface(fn=predict, inputs="image", outputs="text").launch()