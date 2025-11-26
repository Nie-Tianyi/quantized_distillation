import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from matplotlib import pyplot as plt


def download_cifar100():
    print("开始下载CIFAR-100数据集...")

    # 定义数据预处理
    transform_train = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ]
    )

    transform_test = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ]
    )

    # 下载训练集
    print("下载训练集中...")
    trainset = torchvision.datasets.CIFAR100(
        root=".././data",
        train=True,  # 这是训练集
        download=True,
        transform=transform_train,
    )

    # 下载测试集
    print("下载测试集中...")
    testset = torchvision.datasets.CIFAR100(
        root=".././data",
        train=False,  # 这是测试集
        download=True,
        transform=transform_test,
    )

    # 创建数据加载器
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=128, shuffle=True, num_workers=2
    )

    testloader = torch.utils.data.DataLoader(
        testset, batch_size=100, shuffle=False, num_workers=2
    )

    print("下载完成！")
    print(f"训练集样本数: {len(trainset)}")
    print(f"测试集样本数: {len(testset)}")
    print(f"类别数: {len(trainset.classes)}")

    return trainset, testset, trainloader, testloader


def visualize_dataset():
    """可视化一些样本来检查数据集"""
    trainset, testset, trainloader, testloader = download_cifar100()

    # 获取一个batch的数据
    dataiter = iter(trainloader)
    images, labels = next(dataiter)

    # CIFAR-100的类别名称
    classes = trainset.classes

    print(f"图像维度: {images.shape}")  # 应该是 [128, 3, 32, 32]
    print(f"标签维度: {labels.shape}")  # 应该是 [128]

    # 显示前6张图片
    fig, axes = plt.subplots(2, 3, figsize=(10, 6))
    for i in range(6):
        ax = axes[i // 3, i % 3]
        # 反归一化显示
        img = images[i] / 2 + 0.5  # 反归一化
        npimg = img.numpy()
        ax.imshow(np.transpose(npimg, (1, 2, 0)))
        ax.set_title(f"Label: {classes[labels[i]]}")
        ax.axis("off")

    plt.tight_layout()
    plt.savefig("cifar100_samples.png", dpi=150)
    plt.show()


def dataset_statistics():
    """打印数据集的详细统计信息"""
    trainset, testset, trainloader, testloader = download_cifar100()

    print("=" * 50)
    print("CIFAR-100 数据集统计信息")
    print("=" * 50)
    print(f"训练集大小: {len(trainset)} 张图片")
    print(f"测试集大小: {len(testset)} 张图片")
    print(f"图片尺寸: 32x32 RGB")
    print(f"类别数量: {len(trainset.classes)}")
    print(f"批量大小: 128 (训练), 100 (测试)")

    # 统计每个类别的样本数
    train_labels = [label for _, label in trainset]
    test_labels = [label for _, label in testset]

    print(f"\n训练集类别分布:")
    unique, counts = np.unique(train_labels, return_counts=True)
    for i, (cls, count) in enumerate(zip(unique, counts)):
        print(f"  {trainset.classes[cls]:20s}: {count:4d} 张图片")
        if i >= 5:  # 只显示前5个类别
            print(f"  ... (还有 {len(unique) - 5} 个类别)")
            break

    print(f"\n数据加载器信息:")
    print(f"  训练批次数: {len(trainloader)}")
    print(f"  测试批次数: {len(testloader)}")


# 执行下载
if __name__ == "__main__":
    trainset, testset, trainloader, testloader = download_cifar100()
    # 运行可视化
    visualize_dataset()
    # 运行统计
    dataset_statistics()
