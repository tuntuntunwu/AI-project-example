import matplotlib.pyplot as plt
from PIL import Image
import os

img = "./dataset/mason/mason_1.jpg"
txt = "./dataset/mason/mason_1.txt"

image = Image.open(img).convert('RGB')
plt.imshow(image)
ax = plt.gca()

with open(txt, 'r', encoding='utf-8') as f:
    for line in f.readlines():
        paras = line[:-1].split(' ')
        x_center, y_center, width, height = [float(para) for para in paras[1:5]]
        # 默认框的颜色是黑色，第一个参数是左上角的点坐标，第二个参数是宽，第三个参数是长
        ax.add_patch(plt.Rectangle(((x_center - width / 2) * image.width, (y_center - height / 2) * image.height), width * image.width, height * image.height, color="red", fill=False, linewidth=1))

plt.show()
