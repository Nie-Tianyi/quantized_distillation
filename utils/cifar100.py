import torch
import torchvision
import torchvision.transforms as transforms


class CIFAR100Data:
    """CIFAR-100数据管理类"""

    def __init__(self, batch_size=128, data_dir=".././data"):
        self.batch_size = batch_size
        self.data_dir = data_dir
        self.classes = None
        self.load_data()

    def load_data(self):
        """加载数据集"""
        # 数据增强和归一化
        transform_train = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)
                ),
            ]
        )

        transform_test = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)
                ),
            ]
        )

        # 下载数据集
        self.trainset = torchvision.datasets.CIFAR100(
            root=self.data_dir, train=True, download=True, transform=transform_train
        )

        self.testset = torchvision.datasets.CIFAR100(
            root=self.data_dir, train=False, download=True, transform=transform_test
        )

        self.classes = self.trainset.classes

        # 创建数据加载器
        self.trainloader = torch.utils.data.DataLoader(
            self.trainset, batch_size=self.batch_size, shuffle=True, num_workers=2
        )

        self.testloader = torch.utils.data.DataLoader(
            self.testset, batch_size=100, shuffle=False, num_workers=2
        )

    def get_dataloaders(self):
        """返回数据加载器"""
        return self.trainloader, self.testloader

    def get_datasets(self):
        """返回数据集"""
        return self.trainset, self.testset

    def get_class_names(self):
        """返回类别名称"""
        return self.classes


# 使用示例
if __name__ == "__main__":
    # 创建数据管理器
    data_manager = CIFAR100Data(batch_size=128)

    # 获取数据加载器
    trainloader, testloader = data_manager.get_dataloaders()

    # 获取类别信息
    classes = data_manager.get_class_names()
    print(f"数据集包含 {len(classes)} 个类别")
    print("前10个类别:", classes[:10])
