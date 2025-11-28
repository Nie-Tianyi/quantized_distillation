import torch
import torchvision
import torchvision.transforms as transforms
from torch import optim, nn
from tqdm import tqdm

from utils.criterion import test_model
from utils.micro_resnet import MicroResNet

model_path = "../new_model_weights/microresnet_cifar10_best.pth"

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),  # 随机裁剪
    transforms.RandomHorizontalFlip(),      # 随机水平翻转
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))  # CIFAR10实际均值方差
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

# 下载数据集
trainset = torchvision.datasets.CIFAR10(
    root='../data', train=True, download=True, transform=transform_train
)
testset = torchvision.datasets.CIFAR10(
    root='../data', train=False, download=True, transform=transform_test
)

# 数据加载器
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

resnet_20 = MicroResNet(num_classes=10).to(device)

optimizer = optim.SGD(resnet_20.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[80, 120, 160], gamma=0.1)  # 适应200个epoch
# scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

criterion = nn.CrossEntropyLoss()

def train_model(epochs=100):
    best_acc = 0
    for epoch in range(epochs):
        resnet_20.train()
        running_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(trainloader, desc=f"Epoch {epoch+1}/{epochs}")
        for inputs, targets in pbar:
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = resnet_20(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            pbar.set_postfix({
                "Loss": f"{loss.item():.3f}",
                "Acc": f"{100.*correct/total:.2f}%"
            })

        # 每10个epoch测试一次
        if (epoch + 1) % 10 == 0 or epoch == 0:
            test_acc = test_model(resnet_20, testloader, device)
            print(f"Epoch {epoch+1}: Test Accuracy = {test_acc:.2f}%")

            if test_acc > best_acc:
                best_acc = test_acc
                torch.save(resnet_20.state_dict(), model_path)
                print(f"✅ 新的最佳准确率: {best_acc:.2f}%")

        scheduler.step()

if __name__ == '__main__':
    # 开始训练
    train_model(epochs=100)