
def process_new_data():

    learning_rate = 0.001
    training_epochs = 2
    batch_size = 100

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: 1 - x),  # 이미지 반전
        transforms.Normalize((0.5,), (0.5,))
    ])
    
        #데이터셋: mnist + 추가로 들어온 데이터
    new_dataset = torch.utils.data.TensorDataset(torch.stack(new_images), torch.tensor(new_labels))
 
    mnist_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)

    # MNIST 데이터셋과 새 데이터셋 결합
    combined_dataset = ConcatDataset([mnist_dataset, new_dataset])
    
    train_size = int(0.8 * len(combined_dataset))
    val_size = len(combined_dataset) - train_size
    train_dataset, val_dataset = random_split(combined_dataset, [train_size, val_size])

    #데이터 로더 생성
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)
    
    #모델 로드 
    model_path = '/Users/alookso/1010LMpj/ml-project-ml-pjt-7/saved_model.pth'
    model = load_model(model_path)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
   
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    # 학습 루프
    model.train()
    for epoch in range(training_epochs):
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            if batch_idx % 100 == 0:
                print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)}'
                      f' ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')

    # 모델 저장
    torch.save(model.state_dict(), model_path)
    print(f'Model trained and saved as {model_path}')

    return model_path  # 학습된 모델의 경로 반환


    '''
def preprocessing_new_data():
    new_images_folder = '/Users/alookso/1010LMpj/ml-project-ml-pjt-7/saved_images'
    new_images = []
    new_labels = []

    # 정규화만 수행하는 transform 정의 -> 아래같은 에러가 떠서
    # {model_dag.py:62} ERROR - Error processing 5.png: pic should be PIL Image or ndarray. Got <class 'torch.Tensor'>
    
    transform = transforms.Normalize((0.5,), (0.5,))

    for filename in os.listdir(new_images_folder):
        if filename.endswith('.png'):
            try:
                image_path = os.path.join(new_images_folder, filename)
                label = int(os.path.splitext(filename)[0])  # 파일명에서 레이블 추출
                
                #
                image_array = preprocess_image_file(image_path)
                
                # numpy 배열을 PyTorch 텐서로 변환
                image_tensor = torch.from_numpy(image_array).float()
                
                # 차원 순서 변경 (필요한 경우)
                if image_tensor.shape[0] != 1:
                    image_tensor = image_tensor.permute(2, 0, 1)
                
                # 정규화 적용
                image_tensor = transform(image_tensor)

                new_images.append(image_tensor)
                new_labels.append(label)
                
                logging.info(f"Processed image: {filename}")
            except Exception as e:
                logging.error(f"Error processing {filename}: {str(e)}")
'''