# model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import torch.optim as optim
import os
from monitoring import log_info

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 랜덤 시드 고정
torch.manual_seed(2024)
if device == 'cuda':
    torch.cuda.manual_seed_all(2024)

class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.keep_prob = 0.5

        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1)
        )

        self.fc1 = nn.Linear(4 * 4 * 128, 625, bias=True)
        nn.init.xavier_uniform_(self.fc1.weight)
        self.layer4 = nn.Sequential(
            self.fc1,
            nn.ReLU(),
            nn.Dropout(p=1 - self.keep_prob)
        )

        self.fc2 = nn.Linear(625, 10, bias=True)
        nn.init.xavier_uniform_(self.fc2.weight)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)  # 수정된 부분
        out = self.layer3(out)
        out = out.view(-1, 4 * 4 * 128)
        out = self.layer4(out)
        out = self.fc2(out)
        return out

def load_model(model_path='saved_model.pth'):
    model = CNNModel().to(device)
    if not os.path.exists(model_path):
        print("Model file not found. Training a new model...")
        train_and_evaluate_model()
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model

def predict(model, input_data_list):
    predictions = []
    with torch.no_grad():
        for idx, input_data in enumerate(input_data_list):
            input_tensor = torch.from_numpy(input_data).float().to(device)
            input_tensor = input_tensor.unsqueeze(0)  # 배치 차원 추가
            outputs = model(input_tensor)
            _, predicted = torch.max(outputs.data, 1)
            predictions.append(predicted.item())
    return predictions

def train_and_evaluate_model():
    learning_rate = 0.001
    training_epochs = 15
    batch_size = 100

    # 데이터 증강을 포함한 변환 정의
    train_transform = transforms.Compose([
        transforms.RandomRotation(10),        # ±10도 회전
        transforms.RandomAffine(0, shear=10, scale=(0.8,1.2)),  # 전단 변환 및 스케일 조정
        transforms.RandomHorizontalFlip(),    # 좌우 반전
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # MNIST 데이터셋 로드
    full_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=train_transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=test_transform)

    # 데이터셋 분할 (80% 학습, 20% 검증)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    model = CNNModel().to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    best_val_accuracy = 0.0

    for epoch in range(training_epochs):
        model.train()
        avg_cost = 0

        for X, Y in train_loader:
            X = X.to(device)
            Y = Y.to(device)

            optimizer.zero_grad()
            hypothesis = model(X)
            cost = criterion(hypothesis, Y)
            cost.backward()
            optimizer.step()

            avg_cost += cost / len(train_loader)

        print(f'[Epoch: {epoch + 1:>4}] cost = {avg_cost:.9f}')

        # 검증 단계
        model.eval()
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_accuracy = 100 * val_correct / val_total
        print(f'Validation Accuracy: {val_accuracy:.2f}%')

        # 가장 좋은 모델 저장
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), 'saved_model.pth')
            print(f'Best model saved with Validation Accuracy: {val_accuracy:.2f}%')

    print("Training complete.")

    # 테스트 데이터로 모델 평가
    test_accuracy = evaluate_model(model, test_loader)
    print(f'Test Accuracy: {test_accuracy:.2f}%')

def evaluate_model(model, dataloader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
    accuracy = 100 * correct / total
    return accuracy

if __name__ == "__main__":
    train_and_evaluate_model()