from torch.utils.data import Dataset
import os
from PIL import Image


class MyData(Dataset):

    def __init__(self, root_dir, label_dir):
        self.root_dir = root_dir  # 获取到my_dataset/hymenoptera_data/train
        self.label_dir = label_dir  # 获取train下面的文件ants
        self.path = os.path.join(self.root_dir, self.label_dir)  # 将上面两个路径进行连接
        self.img_path = os.listdir(self.path)  # 获取ants下面的全部文件， 这里就是获取全部的图片

    def __getitem__(self, idx):  # 获取图片
        img_name = self.img_path[idx]  # 获取图片名称
        img_item_path = os.path.join(self.root_dir, self.label_dir, img_name)  # 连接成图片的路径
        img = Image.open(img_item_path)  # 获取图片详细内容
        label = self.label_dir
        return img, label

    def __len__(self):
        return len(self.img_path)


root_dir = 'my_dataset/hymenoptera_data/train'
ant_label_dir = 'ants'
bees_label_dir = 'bees'
ant_data = MyData(root_dir, ant_label_dir)
bee_data = MyData(root_dir, bees_label_dir)

train_set = ant_data + bee_data