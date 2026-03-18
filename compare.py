import torch
import torch.nn as nn
from model import PneumoniaCNN
from optimizer import LipschitzMomentum
from utils import get_data

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_loader, _ = get_data()

def run(opt_type):
    model = PneumoniaCNN().to(device)
    criterion = nn.CrossEntropyLoss()

    if opt_type == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    else:
        optimizer = LipschitzMomentum(model.parameters(), lr=0.001)

    for epoch in range(2):
        total_loss = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(opt_type, "Epoch", epoch+1, "Loss:", total_loss)

run("custom")
run("adam")