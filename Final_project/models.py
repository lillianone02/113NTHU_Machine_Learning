import torch
import torch.nn as nn
import torchvision.models as models

def get_regnet(num_classes):
    model = models.regnet_y_400mf(pretrained=True)

    # 解凍2層
    for name, param in model.named_parameters():
        if "trunk_output.block1" in name or "trunk_output.block2" in name:
            param.requires_grad = False

    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)

    return model

def get_resnet50(num_classes):
    model = models.resnet50(pretrained=True)

    # 凍結所有層的參數
    for param in model.parameters():
        param.requires_grad = False

    # 替換最後的全連接層
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)

    # 解凍最後的全連接層
    for param in model.fc.parameters():
        param.requires_grad = True

    return model

def get_efficientnet_v2(num_classes):
    model = models.efficientnet_v2_s(pretrained=True)

    # 凍結所有層的參數
    for param in model.parameters():
        param.requires_grad = False

    # 替換最後的分類層
    num_ftrs = model.classifier[1].in_features  # EfficientNetV2 的分類層是 Sequential，索引 1 是全连接层
    model.classifier[1] = nn.Linear(num_ftrs, num_classes)

    # 解凍最後的全連接層
    for param in model.classifier[1].parameters():
        param.requires_grad = True

    return model

def get_swin_b_vit(num_classes):
    model = models.swin_b(pretrained=True)

    # 凍結所有層的參數
    for param in model.parameters():
        param.requires_grad = False

    # 替換最後的分類頭
    num_ftrs = model.head.in_features
    model.head = nn.Linear(num_ftrs, num_classes)

    # 解凍最後的全連接層
    for param in model.head.parameters():
        param.requires_grad = True

    return model

def get_convnext(num_classes):
    model = models.convnext_base(pretrained=True)

    # 凍結所有層的參數
    for param in model.parameters():
        param.requires_grad = False

    # 替換最後的分類層
    num_ftrs = model.classifier[2].in_features  # ConvNeXt 的分類層是 Sequential，索引 2 是全连接层
    model.classifier[2] = nn.Linear(num_ftrs, num_classes)

    # 解凍最後的全連接層
    for param in model.classifier[2].parameters():
        param.requires_grad = True

    return model
