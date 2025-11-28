import torch
import torchvision
from torch import optim
from torchvision import transforms
from tqdm import tqdm

from utils.criterion import test_model
from utils.distill_loss import FeatureAdaptiveDistillation
from utils.micro_resnet import MicroResNet
from utils.res_net import ResNet20


def feature_adaptive_distillation():
    """特征适配蒸馏训练"""

    model_path = "../new_model_weights/feature_adaptive_distillation_result.pth"

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

    # 1. 初始化模型
    teacher_model = ResNet20(num_classes=10)
    student_model = MicroResNet(num_classes=10)
    # student_model.load_state_dict(
    #     torch.load(model_path)
    # )

    # 加载预训练教师
    teacher_model.load_state_dict(torch.load('../new_model_weights/resnet20_cifar10_best.pth'))

    # 2. 获取特征维度
    def get_dims():
        dummy = torch.randn(2, 3, 32, 32)
        with torch.no_grad():
            _, t_feat = teacher_model(dummy, return_feature=True)
        _, s_feat = student_model(dummy, return_features=True)
        return t_feat.shape[1], s_feat.shape[1]

    teacher_dim, student_dim = get_dims()
    print(f"特征维度: 教师={teacher_dim}, 学生={student_dim}")

    # 3. 初始化特征适配蒸馏
    criterion = FeatureAdaptiveDistillation(
        teacher_dim=teacher_dim,
        student_dim=student_dim,
        temperature=4
    )

    # 4. 优化器（同时优化学生模型和适配器）
    optimizer = optim.Adam(
        list(student_model.parameters()) + list(criterion.adapter.parameters()),
        lr=0.001
    )

    # 5. 训练循环
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    teacher_model.to(device)
    student_model.to(device)
    criterion.to(device)

    teacher_model.eval()  # 固定教师模型

    best_acc = 0
    total_epoch = 200

    for epoch in range(total_epoch):
        student_model.train()
        total_loss = 0
        correct = 0
        total = 0

        pbar = tqdm(trainloader, desc=f"Student KD Epoch {epoch + 1}/{total_epoch}")
        for batch_idx, (inputs, targets) in enumerate(pbar):
            inputs, targets = inputs.to(device), targets.to(device)

            # 教师推理
            with torch.no_grad():
                teacher_logits, teacher_features = teacher_model(inputs, return_feature=True)

            # 学生推理
            student_logits, student_features = student_model(inputs, return_features=True)

            # 计算特征适配蒸馏损失
            loss = criterion(
                student_logits=student_logits,
                teacher_logits=teacher_logits,
                student_feat=student_features,
                teacher_feat=teacher_features,
                targets=targets,
                alpha=0.7
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = student_logits.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            pbar.set_postfix({
                "Loss": f"{loss.item():.3f}",
                "Acc": f"{100.*correct/total:.2f}%"
            })

        avg_loss = total_loss / len(trainloader)

        # 测试
        if (epoch + 1) % 10 == 0:
            test_acc = test_model(student_model, testloader, device)
            print(f'Test Accuracy: {test_acc:.2f}%')

            if test_acc > best_acc:
                best_acc = test_acc
                torch.save(student_model.state_dict(), model_path)
                print(f"✅ 新的最佳准确率: {best_acc:.2f}%")


# 运行训练
if __name__ == '__main__':
    feature_adaptive_distillation()