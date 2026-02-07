import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split

from model import GestureNet

X = np.load("data/X.npy")
Y = np.load("data/Y.npy")

X = torch.tensor(X, dtype=torch.float32)
Y = torch.tensor(Y, dtype=torch.long)

X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, random_state=42)

model = GestureNet()

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(50):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs, Y_train)
    loss.backward()
    optimizer.step()

    model.eval()
    with torch.no_grad():
        val_pred = model(X_val)
        val_acc = (val_pred.argmax(1) == Y_val).float().mean()

    print(f"Epoch {epoch+1} | loss={loss.item():.4f} | val_acc={val_acc:.3f}")

torch.save(model.state_dict(), "gesture.pth")
print("Model saved to gesture.pth")
