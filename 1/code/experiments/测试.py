import json
import numpy as np

from torch.nn import Module, Conv2d, MaxPool2d, Linear, BatchNorm2d, ReLU, Sequential
from torch import load, no_grad, from_numpy, reshape, sort
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from PIL import Image


# VGG
cfgs = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

class VGG(Module):
    def __init__(self, vgg_name, class_number):
        super(VGG, self).__init__()
        # features
        self.features = self._make_layers(cfgs[vgg_name])
        # classifier
        #self.classifier = nn.Linear(4096, 249)
        self.classifier = Linear(8192, class_number)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [Conv2d(in_channels, x, kernel_size=3, padding=1),
                           BatchNorm2d(x),  # BatchNorm2d() is helpful
                           ReLU(inplace=True)]
                in_channels = x
        return Sequential(*layers)

def jinwenPredict(PIL_RGB_img, dict_path, model_path):
    with open(dict_path, encoding='utf-8') as f:
        classes = json.load(f)
        classes = tuple(classes)
    class_number = len(classes)
    # print(classes)
    # print(class_number)

    # predict
    # load net
    net = VGG('VGG11', class_number)
    net.load_state_dict(load(model_path, map_location='cpu'))
    net.eval()

    with no_grad():
        # preprocess image data
        transform = Compose([
            Resize((128, 128)),
            ToTensor(),
            Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        img = transform(PIL_RGB_img)

        #image = img.numpy().reshape(-1, 3, 64, 64)
        image = img.reshape(-1, 3, 128, 128)
        outputs = net(image)

        result = outputs[0]
        # use softmax to get probability
        values = result.numpy().copy()
        probs = np.exp(values) / np.sum(np.exp(values))
        predicted_top5 = np.argsort(probs)[-5:]

        return classes[predicted_top5[-1]], round(probs[predicted_top5[-1]] * 100, 2),\
            classes[predicted_top5[-2]], round(probs[predicted_top5[-2]] * 100, 2),\
            classes[predicted_top5[-3]], round(probs[predicted_top5[-3]] * 100, 2),\
            classes[predicted_top5[-4]], round(probs[predicted_top5[-4]] * 100, 2),\
            classes[predicted_top5[-5]], round(probs[predicted_top5[-5]] * 100, 2)

def characterRecognition(img_path, dict_path, model_path):
    # transform to RGB images having 3 channels
    img0 = Image.open(img_path).convert('RGB')
    result0 = jinwenPredict(img0, dict_path, model_path)

    # PIL.Image to numpy.ndarray
    img = np.array(Image.open(img_path))
    img = 255 -img
    # numpy.ndarray to PIL.Image
    img1 = Image.fromarray(img.astype('uint8')).convert('RGB')
    result1 = jinwenPredict(img1, dict_path, model_path)
    
    if result0[1] >= result1[1]:
        return result0
    else:
        return result1
        

if __name__ == '__main__':
    # test
    print(characterRecognition(r"E:\Code\jinwen-recognition\v3\data\2.select+group\1001-\亞\02_00340_001_A.jpg",
        "./result/hf-words/classes.json", "./result/hf-words/itr8_jinwen_vgg11.pt"))
