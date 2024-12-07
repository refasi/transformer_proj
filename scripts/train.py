import torch
from torch import nn, optim
from utils.data_loader import load_data
from models.lstm import LSTMModel

def train_model():
    data_loader = load_data()
    model = LSTMModel(input_size=100, hidden_size=128, output_size=1, num_layers=2)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Training loop
    for epoch in range(20):
        for inputs, targets in data_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}, Loss: {loss.item()}")
