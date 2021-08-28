import torchvision
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

test_data = torchvision.datasets.CIFAR10(
    root='./dataset',
    train=False,
    transform=torchvision.transforms.ToTensor(),
    download=True
)

test_load = DataLoader(
    dataset=test_data,
    shuffle=True,
    drop_last=True,
    num_workers=0,
    batch_size=64
)

step = 0
writer = SummaryWriter('./dataset')
for data in test_load:
    imgs, targets = data
    writer.add_images('test_load_set', imgs, step)
    step = step + 1

writer.close()