def Cal_Accuarcy(model):
    correct = 0
    total = 0
    for data in test_loader:
        images, label = data
        images, label = images.to(device), labels.to(device)
        outputs = model(images)
        _, predictions = torch.max(outputs.data, 1)
        total += label.size(0)
        correct += (predictions == label).sum().item()
    return 100 * correct / total