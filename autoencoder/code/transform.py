import torch
import torchvision
from torchvision import datasets, transforms

# 분류된 이미지가 있는 폴더의 경로
data_path = './workspace_project/9oz/A/17_sorted'

# 폴더 내의 이미지를 불러오고 변형하는 코드
custom_dataset = datasets.ImageFolder(
    root=data_path,
    transform=transforms.Compose([
        transforms.Resize((299, 299)),#600*800
        transforms.ToTensor()         # Tensor로 변환  original images are already normalized to the range [0, 1] due to the use of transforms.ToTensor()
    ])
)

# DataLoader 설정
BATCH_SIZE = 64  # 적절한 배치 크기로 설정하세요
custom_loader = torch.utils.data.DataLoader(
    dataset=custom_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=3
)
