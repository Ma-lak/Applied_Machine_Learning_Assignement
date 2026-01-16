import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
from sklearn.metrics import precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
from ModelB.resnet18 import ResNet18


def resnet18_train_evaluate(trainloader, testloader, valloader):
    """
    Train and evaluate ResNet18 on input images.

    Args:
        trainloader (DataLoader): DataLoader for the training set.
        testloader (DataLoader): DataLoader for the test set.
        valloader (DataLoader): DataLoader for the validation set.

    Returns:
        model: Trained ResNet18 model
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Define the model
    model = ResNet18().to(device)



    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
   # optimizer = optim.SGD(model.parameters(), lr=0.3, momentum=0.9, weight_decay=5e-4) # may change back again to 0.3
   #scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[5,7], gamma=0.1) 
    num_epochs = 10 
    train_losses, train_acc_list, test_acc_list, val_acc_list = [], [], [], []

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct, total = 0, 0
        
        for inputs, labels in trainloader:
          inputs, labels = inputs.to(device), labels.to(device)
          optimizer.zero_grad()
          outputs = model(inputs)
          loss = criterion(outputs, labels)
          loss.backward()
          optimizer.step()

          running_loss += loss.item() * inputs.size(0)
          _, predicted = outputs.max(1)
          total += labels.size(0)
          correct += predicted.eq(labels).sum().item()
        train_loss = running_loss / len(trainloader.dataset)
        train_acc = 100. * correct / total
        train_losses.append(train_loss)
        train_acc_list.append(train_acc)
        model.eval()
        correct, total = 0, 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for inputs, labels in valloader:
                inputs= inputs.to(device)
                labels = labels.view(-1).long().to(device)
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        val_acc = 100. * correct / total
        val_acc_list.append(val_acc)
        print(f'Epoch [{epoch+1}/{num_epochs}] '
              f'Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%')      
        
        with torch.no_grad():
         for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            all_preds.append(predicted.cpu())
            all_labels.append(labels.cpu())
        test_acc = 100. * correct / total
        test_acc_list.append(test_acc)
        all_preds = torch.cat(all_preds).numpy()
        all_labels = torch.cat(all_labels).numpy()
        test_precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
        test_recall    = recall_score(all_labels, all_preds, average='macro', zero_division=0)
        test_f1        = f1_score(all_labels, all_preds, average='macro', zero_division=0)
        scheduler.step()
        optimizer.zero_grad()
        print(f"Test Acc: {test_acc:.2f}% | "
         f"Precision: {test_precision:.4f} | "
         f"Recall: {test_recall:.4f} | "
         f"F1: {test_f1:.4f}")

    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    plt.plot(train_losses, label='Train Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.legend()

    plt.subplot(1,2,2)
    plt.plot(train_acc_list, label='Train Accuracy')
    plt.plot(test_acc_list, label='Test Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.title('Accuracy')
    plt.legend()

    plt.show()
    return model

