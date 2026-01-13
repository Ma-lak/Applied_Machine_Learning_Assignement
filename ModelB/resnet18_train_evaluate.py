import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
from ModelB.resnet18 import ResNet18
import matplotlib.pyplot as plt

def resnet18_train_evaluate(trainloader, testloader, valloader):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Define the model
    model = ResNet18().to(device)
    #print(model)


    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.3, momentum=0.9, weight_decay=5e-4) # may change back again to 0.003 or 0.3
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

    num_epochs = 10 # maybe keep 6 for augment, same acc but less spikes (do two different num_epochs)
    train_losses, train_acc_list, test_acc_list = [], [], []
    
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
        
        with torch.no_grad():
         for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        test_acc = 100. * correct / total
        test_acc_list.append(test_acc)
        scheduler.step()
        print(f'Epoch [{epoch+1}/{num_epochs}] Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | Test Acc: {test_acc:.2f}%')
        print('Accuracy of the network: %d %%' % (100 * correct / total))


        optimizer.zero_grad()

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

