"""使用知识蒸馏训练学生模型"""
import torch
import torchvision
from torch import optim
from torchvision.transforms import transforms
from tqdm import tqdm

from utils.criterion import test_model
from utils.distill_loss import DistillationLoss
from utils.micro_ghost_resnet import MicroResNetGhost
from utils.persist import persist_learning_history
from utils.res_net import ResNet20


def train_microresnet_ghost_with_distillation():
    print("🚀 开始训练学生模型 MicroResNetGhost（带知识蒸馏）...")

    model_path = "../new_model_weights/microresnet_ghost_cifar10_distill_best.pth"

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


    # 模型和优化器
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    teacher_model = ResNet20(num_classes=10).to(device)
    student_model = MicroResNetGhost(num_classes=10).to(device)

    # 加载预训练的教师模型
    teacher_model.load_state_dict(torch.load("../new_model_weights/resnet20_cifar10_best.pth"))
    teacher_model.eval()
    for param in teacher_model.parameters():
        param.requires_grad = False

    optimizer = optim.SGD(
        student_model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4
    )
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[80, 120, 160], gamma=0.1)
    criterion = DistillationLoss(alpha=0.7, temperature=4)

    # 训练记录
    best_acc = 0
    loss_history = []

    # 训练循环
    for epoch in range(200):
        student_model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(trainloader, desc=f"Student KD Epoch {epoch + 1}/200")
        for batch_idx, (inputs, targets) in enumerate(pbar):
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()

            # 学生模型输出
            student_outputs = student_model(inputs)

            # 教师模型输出（不计算梯度）
            with torch.no_grad():
                teacher_outputs = teacher_model(inputs)

            # 计算蒸馏损失
            loss = criterion(student_outputs, teacher_outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = student_outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            pbar.set_postfix({
                "Loss": f"{loss.item():.3f}",
                "Acc": f"{100.*correct/total:.2f}%"
            })

        # 记录损失
        loss_history.append(running_loss / len(trainloader))

        # 每10个epoch测试一次
        if (epoch + 1) % 10 == 0 or epoch == 0:
            test_acc = test_model(student_model, testloader, device)
            print(f"Epoch {epoch+1}: Test Accuracy = {test_acc:.2f}%")

            if test_acc > best_acc:
                best_acc = test_acc
                torch.save(student_model.state_dict(), model_path)
                print(f"✅ 新的最佳准确率: {best_acc:.2f}%")

        scheduler.step()

    persist_learning_history(loss_history, "microresnet_ghost_loss")


if __name__ == '__main__':
    train_microresnet_ghost_with_distillation()
