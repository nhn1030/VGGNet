import torch
import torchvision
import torchvision.transforms as transforms

def get_train_loader(root, batch_size, num_workers):
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    
    train_dataset = torchvision.datasets.CIFAR10(
        root= root, train=True, download=True, transform=transform_train
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )

    return train_loader

def get_test_loader(root, batch_size, num_workers):
    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    test_dataset = torchvision.datasets.CIFAR10(
        root= root, train=False, download=True, transform=transform_test
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    return test_loader


root = 'datasets'  # 실제 데이터셋이 위치한 디렉토리 경로로 수정해주세요.
batch_size = 64
num_workers = 2

train_loader = get_train_loader(root, batch_size, num_workers)
test_loader = get_test_loader(root, batch_size, num_workers)
# 데이터셋 사용 예시
# for images, labels in train_loader:
#     # 학습 데이터셋 사용
#     # images: [batch_size, 3, 32, 32] 크기의 이미지 배치
#     # labels: [batch_size] 크기의 정답 레이블 배치
#     pass

# for images, labels in test_loader:
#     # 테스트 데이터셋 사용
#     pass

# train_loader = train_loader(batch_size=128, num_workers=2)
# test_loader = test_loader(batch_size=128, num_workers=2)