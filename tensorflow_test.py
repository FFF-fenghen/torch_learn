from torch.utils.tensorboard import SummaryWriter
from PIL import Image
import numpy as np

writer = SummaryWriter('./logs')  # 把信息写到目标文件 logs 中

img_path = 'my_dataset/hymenoptera_data/train/ants_img/6240338_93729615ec.jpg'
img = Image.open(img_path)  # 打开图片并获取相应信息
img_array = np.array(img)  # 将图片数据类型转换成numpy数据类型
writer.add_image('tag', img_array, 2, dataformats='HWC')
# for i in range(100):  # 绘制标量数据的函数
#     writer.add_scalar('y=2x', 2*i, i)

writer.close()
