# 主要学习trnasform最主要的几种方式
# 输入  PIL       Image.open()
# 输出  tensor    ToTensor()
# 作用  narrays   cv.Imread()

from torchvision import transforms
from PIL import Image
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter('./logs')
img_path = 'my_dataset/hymenoptera_data/train/ants_img/6240338_93729615ec.jpg'
img_PIL = Image.open(img_path)
tensor_object = transforms.ToTensor()
img_tensor = tensor_object(img_PIL)
writer.add_image('img_tensor', img_tensor)

# normalize 归一化，改变图片的颜色，但是用途不明
trans_norm = transforms.Normalize([5, 8, 3], [4, 8, 5])
img_norm = trans_norm(img_tensor)
writer.add_image('tensor_norm', img_norm, 1)

# Resize
# 设置目标图片的尺寸
tensor_resize = transforms.Resize((512, 512))
# img PIL -> img PIL
img_tensor = tensor_resize(img_PIL)
# img PIL -> img tensor
img_tensor = tensor_object(img_tensor)
writer.add_image('resize', img_tensor, 0)

writer.close()
