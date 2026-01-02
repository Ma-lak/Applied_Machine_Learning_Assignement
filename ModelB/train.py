import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import resnet18
from sklearn.metrics import accuracy_score

def train_evaluate_resnet18(X_train, y_train, X_val, y_val, X_test, y_test, config):
    device = torch.device(config.get("device", "cpu"))

    net = resnet18(pretrained=True)
    net.fc = nn.Linear(net.fc.in_features, config["num_classes"])
    net = net.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=config.get("lr", 1e-3))

    losses = []
    val_accuracies = []

    for epoch in range(10):
        net.train()

        optimizer.zero_grad()
        outputs = net(X_train.to(device))
        loss = criterion(outputs, y_train.to(device))
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

        # Validation
        net.eval()
        with torch.no_grad():
            val_outputs = net(X_val.to(device))
            val_preds = torch.argmax(val_outputs, dim=1)
            val_acc = accuracy_score(y_val.cpu().numpy(), val_preds.cpu().numpy())
            val_accuracies.append(val_acc)

        if (epoch + 1) % config.get("print_every", 1) == 0:
            print(f"Epoch [{epoch+1}], Loss: {loss.item():.4f}, Val Acc: {val_acc:.4f}")

    # Test evaluation
    net.eval()
    with torch.no_grad():
        test_outputs = net(X_test.to(device))
        test_preds = torch.argmax(test_outputs, dim=1)
        test_accuracy = accuracy_score(y_test.cpu().numpy(), test_preds.cpu().numpy())

    return {
        "experiment": "ResNet18 ",
        "losses": losses,
        "val_accuracies": val_accuracies,
        "test_accuracy": test_accuracy
    }
