"""使用知识蒸馏训练学生模型"""
import torch
import torchvision
from torch import optim
from torchvision.transforms import transforms
from tqdm import tqdm

from utils.criterion import test_model
from utils.distill_loss import ProgressiveDistillation
from utils.micro_resnet import MicroResNet
from utils.res_net import ResNet20

if __name__ == '__main__':
    print("Progressive Distillation")

    model_path = "../new_model_weights/progressive_distillation_result.pth"

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
    student_model = MicroResNet(num_classes=10).to(device)

    # 加载预训练的教师模型
    teacher_model.load_state_dict(torch.load("../new_model_weights/resnet20_cifar10_best.pth"))
    teacher_model.eval()
    for param in teacher_model.parameters():
        param.requires_grad = False

    optimizer = optim.SGD(
        student_model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4
    )
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[80, 120, 160], gamma=0.1)
    criterion = ProgressiveDistillation(initial_alpha=0.1, final_alpha=0.7, initial_temperature=2, final_temperature=8)

    # 训练记录
    best_acc = 0
    total_epoch = 200

    # 训练循环
    for epoch in range(total_epoch):
        student_model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(trainloader, desc=f"Student KD Epoch {epoch + 1}/{total_epoch}")
        for batch_idx, (inputs, targets) in enumerate(pbar):
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()

            # 学生模型输出
            student_outputs = student_model(inputs)

            # 教师模型输出（不计算梯度）
            with torch.no_grad():
                teacher_outputs = teacher_model(inputs)

            # 计算蒸馏损失
            loss = criterion(student_outputs, teacher_outputs, targets, epoch, total_epoch)
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

        # 每10个epoch测试一次
        if (epoch + 1) % 10 == 0 or epoch == 0:
            test_acc = test_model(student_model, testloader, device)
            print(f"Epoch {epoch+1}: Test Accuracy = {test_acc:.2f}%")

            if test_acc > best_acc:
                best_acc = test_acc
                torch.save(student_model.state_dict(), model_path)
                print(f"✅ 新的最佳准确率: {best_acc:.2f}%")

        scheduler.step()

