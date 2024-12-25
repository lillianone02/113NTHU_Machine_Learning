import os
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from torchvision import models, transforms
from sklearn.model_selection import train_test_split
from detect_face_and_landmarks import detect_face_and_landmarks
from PIL import Image
# 情緒分類與對應的標籤
emotions = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
emotion_to_label = {emotion: idx for idx, emotion in enumerate(emotions)}

# 自定義資料集類別
class EmotionDataset(Dataset):
    def __init__(self, data, labels, transform=None):
        self.data = data
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data[idx]
        image = cv2.imread(img_path, cv2.IMREAD_COLOR)  # 使用彩色圖像
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # 將 BGR 轉換為 RGB，符合 torchvision 的需求
        if image is None:
            raise ValueError(f"Could not read image at {img_path}")
        processed_face = detect_face_and_landmarks(image)
        processed_face_rgb = cv2.cvtColor(processed_face, cv2.COLOR_BGR2RGB)
        if self.transform:
            processed_face_pil = Image.fromarray(processed_face_rgb)
            image = self.transform(processed_face_pil)
        else:
            # 如果沒有 transform，做基本處理
            processed_face_resized  = cv2.resize(processed_face_rgb, (224, 224))
            image = torch.tensor(processed_face_resized, dtype=torch.float32).permute(2, 0, 1) / 255.0
        label = self.labels[idx]
        return image, label

# 數據增強與處理
transform_augment = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),  # 隨機水平翻轉
    transforms.RandomRotation(10),  # 隨機旋轉 ±10 度
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),  # 隨機調整顏色屬性
    transforms.ToTensor(),  # 轉為張量
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 標準化
])

transform_test = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 與 ImageNet 模型預訓練標準化一致
])

# 讀取數據
def load_data(data_dir):
    data = []
    labels = []
    for emotion in emotions:
        emotion_dir = os.path.join(data_dir, emotion)
        label = emotion_to_label[emotion]
        if not os.path.exists(emotion_dir):
            continue
        for img_name in os.listdir(emotion_dir):
            img_path = os.path.join(emotion_dir, img_name)
            data.append(img_path)
            labels.append(label)
    return data, labels

# 加載訓練數據
train_dir = 'your_path_to_data_folder'
train_data, train_labels = load_data(train_dir)

# 加載測試數據
test_dir = 'your_path_to_data_folder'
test_data, test_labels = load_data(test_dir)

# 建立訓練和測試資料集
train_dataset = EmotionDataset(train_data, train_labels, transform=transform_augment)
test_dataset = EmotionDataset(test_data, test_labels, transform=transform_test)

# 分割訓練集為訓練和驗證集
val_size = int(0.1 * len(train_dataset))
train_size = len(train_dataset) - val_size

train_dataset, val_dataset = torch.utils.data.random_split(
    train_dataset, [train_size, val_size])

# 創建數據加載器
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# 確認數據加載正常
print(f"訓練集大小: {len(train_dataset)}")
print(f"驗證集大小: {len(val_dataset)}")
print(f"測試集大小: {len(test_dataset)}")

# 顯示前兩張處理後的影像，從 train_dataset 取樣
plain_dataset = EmotionDataset(train_data, train_labels, transform=None)
os.makedirs("check_landmarks", exist_ok=True)

for i in range(10):
    img, label = plain_dataset[i]  # img 是 Tensor [C,H,W]
    # 將張量轉為 NumPy
    img_np = img.permute(1, 2, 0).numpy()
    img_np = np.clip(img_np, 0, 1)
    img_np = (img_np * 255).astype(np.uint8)

    # img_np 現在是 RGB 格式，若要使用 cv2 儲存，需要 BGR 格式
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    cv2.imwrite(f"check_landmarks/sample_{i}.jpg", img_bgr)

print("已儲存前兩張處理後的影像至 check_landmarks 目錄中。")