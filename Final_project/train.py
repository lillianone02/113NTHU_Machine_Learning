import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold
import seaborn as sns
import matplotlib.pyplot as plt

# 引入必要的组件
from data_preprocess import (
    emotions,
    emotion_to_label,
    EmotionDataset,
    transform_augment,
    transform_test,
    load_data,
)
from models import (
    get_regnet,
    get_resnet50,
    get_efficientnet_v2,
    get_convnext,
    get_swin_b_vit,
)


if __name__ == '__main__':
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_data_dir = 'your_path_to_data_folder' 
    test_data_dir = 'your_path_to_data_folder'    

    train_data, train_labels = load_data(train_data_dir)
    test_data, test_labels = load_data(test_data_dir)

    num_epochs = 20
    batch_size = 64

    train_data = np.array(train_data)
    train_labels = np.array(train_labels)

    test_dataset = EmotionDataset(test_data, test_labels, transform=transform_test)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # k-fold cross-validation
    k_folds = 5
    skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)

    fold_accuracies = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(train_data, train_labels)):
        print(f'Fold {fold + 1}/{k_folds}')

        train_fold_data = train_data[train_idx]
        train_fold_labels = train_labels[train_idx]
        val_fold_data = train_data[val_idx]
        val_fold_labels = train_labels[val_idx]

        train_fold_dataset = EmotionDataset(train_fold_data, train_fold_labels, transform=transform_augment)
        val_fold_dataset = EmotionDataset(val_fold_data, val_fold_labels, transform=transform_test)

        train_loader = DataLoader(train_fold_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_fold_dataset, batch_size=batch_size, shuffle=False)
        
        # init model
        model = get_regnet(num_classes=7)
        # model = get_resnet50(num_classes=7)
        # model = get_efficientnet_v2(num_classes=7)
        # model = get_convnext(num_classes=7)
        # model = get_swin_b_vit(num_classes=7)
        
        model.to(device)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.005)
        
        best_val_accuracy = 0.0
        
        for epoch in range(1, num_epochs + 1):
            print(f"Epoch {epoch}/{num_epochs}")
            model.train()
            running_loss = 0.0
            for batch_idx, (inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(device), targets.to(device)
                targets = targets.long()
                
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            
            model.eval()
            val_loss = 0.0
            correct = 0
            total = 0
            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    targets = targets.long()
                    outputs = model(inputs)
                    val_loss += criterion(outputs, targets).item()
                    _, predicted = torch.max(outputs.data, 1)
                    total += targets.size(0)
                    correct += (predicted == targets).sum().item()
            
            val_accuracy = 100.0 * correct / total
            val_loss /= len(val_loader)
            
            print(f'Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%')
            
            # 保存最佳模型
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                torch.save(model.state_dict(), f'best_model_fold{fold + 1}.pth')
                print("Best model saved for this fold.")
        
        fold_accuracies.append(best_val_accuracy)
        print(f'Best Validation Accuracy for fold {fold + 1}: {best_val_accuracy:.2f}%')

    avg_val_accuracy = np.mean(fold_accuracies)
    print(f'Average Validation Accuracy over {k_folds} folds: {avg_val_accuracy:.2f}%')

    best_fold = np.argmax(fold_accuracies) + 1
    print(f'Best fold is fold {best_fold} with validation accuracy {fold_accuracies[best_fold - 1]:.2f}%')
    model.load_state_dict(torch.load(f'best_model_fold{best_fold}.pth'))

    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            targets = targets.long()
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    test_accuracy = 100.0 * np.sum(np.array(all_preds) == np.array(all_targets)) / len(all_targets)
    report = classification_report(all_targets, all_preds, target_names=emotions)
    cm = confusion_matrix(all_targets, all_preds)

    print(f'Test Accuracy: {test_accuracy:.2f}%')
    print('Classification Report:')
    print(report)

    # plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d',
                xticklabels=emotions, yticklabels=emotions, cmap='Blues')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix on Test Set')
    plt.show()