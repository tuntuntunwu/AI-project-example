from torch.nn import Module, Conv2d, MaxPool2d, Linear, BatchNorm2d, ReLU, Sequential
from torch import load, no_grad, from_numpy, reshape, sort
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from PIL import Image
import numpy as np

# VGG
cfgs = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class VGG(Module):
    def __init__(self, vgg_name):
        super(VGG, self).__init__()
        # features
        self.features = self._make_layers(cfgs[vgg_name])
        # classifier
        #self.classifier = nn.Linear(4096, 249)
        self.classifier = Linear(2048, 249)

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


# classes
classes = ('一', '丁', '三', '上', '下', '不', '且', '丕', '丙', '中', '乃',
           '乊', '之', '乍', '乎', '乙', '九', '事', '二', '于', '五', '井', '亗',
           '亞', '亥', '亯', '人', '令', '以', '仲', '伐', '休', '伯', '余', '侎',
           '侯', '保', '倂', '傂', '傭', '元', '兄', '先', '克', '入', '八', '公',
           '六', '其', '内', '冊', '冬', '冶', '初', '則', '剌', '剥', '匄', '北',
           '十', '午', '南', '即', '卿', '又', '友', '受', '召', '台', '史', '右',
           '司', '各', '吉', '君', '吳', '告', '呚', '周', '命', '唯', '商', '啠',
           '啦', '喜', '四', '壬', '壺', '壽', '夕', '夙', '多', '大', '天', '夫',
           '奠', '女', '好', '姜', '姬', '婦', '子', '孝', '孟', '季', '孫', '它',
           '守', '宗', '官', '室', '宮', '家', '寅', '寶', '尊', '對', '小', '尸',
           '尹', '屯', '工', '左', '己', '巿', '師', '年', '庚', '庫', '康', '廷',
           '廿', '弔', '彊', '彝', '從', '御', '徵', '德', '戈', '戊', '戎', '成',
           '我', '才', '拜', '揚', '揵', '攻', '政', '敢', '敬', '文', '新', '方',
           '旂', '旅', '既', '日', '明', '易', '晉', '曰', '曾', '月', '有', '朕',
           '楚', '樂', '正', '武', '母', '氏', '氒', '永', '灁', '灕', '無', '爲',
           '父', '玄', '王', '生', '用', '田', '甲', '申', '疑', '癸', '百', '皇',
           '盨', '眔', '祀', '祈', '福', '穆', '立', '簋', '義', '羽', '考', '者',
           '臣', '自', '般', '若', '萬', '蔡', '虎', '虢', '行', '衛', '角', '貝',
           '賓', '赤', '車', '辛', '辟', '辭', '辰', '追', '造', '邁', '邑', '邦',
           '郾', '金', '鐘', '鑄', '長', '陳', '隹', '霝', '霸', '頌', '顯', '飤',
           '首', '馬', '鬲', '魚', '魯', '黃', '鼎', '齊', '龏', '龢')


def jinwenPredict(img_path):
    # load net
    net = VGG('VGG11')
    net.load_state_dict(load("./res/Jinwen_vgg_12itr.pt"))
    net.eval()

    with no_grad():
        # transform to RGB images having 3 channels
        img = Image.open(img_path).convert('RGB')
        # preprocess image data
        transform = Compose([
            Resize((64, 64)),
            ToTensor(),
            Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        img = transform(img)

        #image = img.numpy().reshape(-1, 3, 64, 64)
        image = img.reshape(-1, 3, 64, 64)
        outputs = net(image)

        result = outputs[0]
        # use softmax to get probability
        values = result.numpy().copy()
        probs = np.exp(values) / np.sum(np.exp(values))
        predicted_top5 = np.argsort(probs)[-5:]

        return classes[predicted_top5[-1]], classes[predicted_top5[-2]],\
            classes[predicted_top5[-3]], classes[predicted_top5[-4]],\
            classes[predicted_top5[-5]],\
            str(round(probs[predicted_top5[-1]] * 100, 2)),\
            str(round(probs[predicted_top5[-2]] * 100, 2)),\
            str(round(probs[predicted_top5[-3]] * 100, 2)),\
            str(round(probs[predicted_top5[-4]] * 100, 2)),\
            str(round(probs[predicted_top5[-5]] * 100, 2))


if __name__ == '__main__':
    # jinwenPredict("e:/Code/demo/res/单字/一/01_00167_206_A.gif")
    pass
