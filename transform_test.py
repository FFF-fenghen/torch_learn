from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms



# tensor 的数据类型
# transform.totensor 解决的问题：
# transform 如何使用
# tensor的使用场景和使用方式

img_path = 'my_dataset/hymenoptera_data/train/ants_img/6240338_93729615ec.jpg'
img = Image.open(img_path)
tensor_tran = transforms.ToTensor()   # 引入tensor对象，
tensor_img = tensor_tran(img)  # 传入一张图像，返回张量数据， 一般一张图片是3维的

writer = SummaryWriter('./logs')

writer.add_image('tensor_test', tensor_img)
writer.close()