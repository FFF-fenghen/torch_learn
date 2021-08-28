import torchvision
from torch.utils.tensorboard import SummaryWriter

dataset_transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()
])
train_set = torchvision.datasets.CIFAR10(root='./dataset',
                                         train=True,
                                         transform=dataset_transform,
                                         download=True)
test_set = torchvision.datasets.CIFAR10(root='./dataset',
                                         train=False,
                                        transform=dataset_transform,
                                        download=True)
#
# print(test_set[0])
#
# img, target = test_set[0]
# print('img:')
# print(img)
# print('target:')
# print(target)
# print(test_set.train)
# print(test_set.classes)

writer = SummaryWriter('../logs')  # 写入的地址位置
for i in range(10):
    img, target = test_set[i]
    writer.add_image('test_set', img, i)

writer.close()