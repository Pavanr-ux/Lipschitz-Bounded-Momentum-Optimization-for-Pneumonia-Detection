print("STARTED RUNNING")

import torch
import torch.nn as nn
from model import PneumoniaCNN
from optimizer import LipschitzMomentum
from utils import get_data
from sklearn.metrics import accuracy_score

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train():
    train_loader, test_loader = get_data()

    model = PneumoniaCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = LipschitzMomentum(model.parameters(), lr=0.001)

    for epoch in range(3):
        total_loss = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

    torch.save(model.state_dict(), "model_lipschitz.pth")

def evaluate():
    train_loader, test_loader = get_data()
    model = PneumoniaCNN().to(device)
    model.load_state_dict(torch.load("model_lipschitz.pth"))
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)

            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    print("Accuracy:", correct / total)

if __name__ == "__main__":
    train()
    evaluate()