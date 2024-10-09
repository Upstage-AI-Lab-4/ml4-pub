# model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import torch.optim as optim
import torch.nn.init
import os

import mlflow
import mlflow.pytorch  # pytorch 모델을 mlflow로 저장하기 위함

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
        out = self.layer2(out)
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
    # FutureWarning 해결을 위해 weights_only=True 설정
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model

def predict(model, input_data_list):
    predictions = []
    with torch.no_grad():
        for input_data in input_data_list:
            input_tensor = torch.from_numpy(input_data).float().to(device)
            input_tensor = input_tensor.unsqueeze(0)  # 배치 차원 추가
            outputs = model(input_tensor)
            _, predicted = torch.max(outputs.data, 1)
            predictions.append(predicted.item())
    return predictions

def train_and_evaluate_model():
    mlflow.set_tracking_uri('http://127.0.0.1:5000')
    # torch 버전이슈로 autolog 불가 
    # mlflow.pytorch.autolog()
    learning_rate = 0.001
    training_epochs = 2
    batch_size = 100

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: 1 - x),  # 이미지 반전
        transforms.Normalize((0.5,), (0.5,))
    ])

    # MNIST 데이터셋 로드
    full_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)

    # 데이터셋 분할 (80% 학습, 20% 검증)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)

    model = CNNModel().to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    best_val_accuracy = 0.0

    # MLflow experiment 시작
    # ml_exp = mlflow.set_experiment(experiment_name='mnist')
    
    with mlflow.start_run():
        mlflow.log_param('learning_rate', learning_rate)
        mlflow.log_param('epochs', training_epochs)
        mlflow.log_param('batch_size', batch_size)

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
            mlflow.log_metric(f'epoch_cost', avg_cost.item(), step=epoch + 1)

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
            mlflow.log_metric(f'val_accuracy', val_accuracy, step=epoch + 1)

            # 가장 좋은 모델 저장
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                torch.save(model.state_dict(), 'saved_model.pth')
                print(f'Best model saved with Validation Accuracy: {val_accuracy:.2f}%')

        # MLflow에 모델 저장
        mlflow.pytorch.log_model(model, "model")
        print("Training complete.")

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

    # 테스트 데이터로 모델 평가
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: 1 - x),  # 이미지 반전
        transforms.Normalize((0.5,), (0.5,))
    ])

    mnist_test = datasets.MNIST(root='./data', train=False, transform=transform, download=True)
    test_loader = DataLoader(dataset=mnist_test, batch_size=100, shuffle=False)

    model = load_model('saved_model.pth')
    test_accuracy = evaluate_model(model, test_loader)
    print(f'Test Accuracy: {test_accuracy:.2f}%')

    # MLflow에 테스트 정확도 기록
    mlflow.log_metric('test_accuracy', test_accuracy)
