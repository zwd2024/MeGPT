
import os
import json
import re
import numpy as np
from PIL import Image
import torch.utils.data as data
from transformers import BertTokenizer, AutoImageProcessor
from PIL import Image
import cv2
from scipy.ndimage import convolve

# Gabor滤波
def apply_gabor_filter(image, sigma_x, sigma_y, theta, lambd, psi, gamma):
    # 确保图像是 float 类型
    image_tensor = image_tensor.float()
    # 计算频率分量
    u = 1
    v = 1
    t_x = 1 / (2 * np.pi * sigma_x ** 2)
    t_y = 1 / (2 * np.pi * sigma_y ** 2)

    # 创建网格
    channels, rows, cols = image.shape
    [x, y] = np.meshgrid(np.arange(cols), np.arange(rows))

    # 旋转坐标
    x_theta = x * np.cos(theta) + y * np.sin(theta)
    y_theta = -x * np.sin(theta) + y * np.cos(theta)

    # Gabor滤波器公式
    gabor_real = np.exp(-t_x * x_theta ** 2 - t_y * y_theta ** 2) * np.cos(2 * np.pi * x_theta / lambd + psi)
    gabor_imag = np.exp(-t_x * x_theta ** 2 - t_y * y_theta ** 2) * np.sin(2 * np.pi * x_theta / lambd + psi)

    # 应用滤波器
    filtered_real = cv2.filter2D(image, cv2.CV_64F, gabor_real)
    filtered_imag = cv2.filter2D(image, cv2.CV_64F, gabor_imag)

    # 合并实部和虚部
    filtered = filtered_real + 1j * filtered_imag

    return np.abs(filtered)


def gabor_3d(x, y, b, f, theta, phi, sigma_x, sigma_y, sigma_b):
    """
    生成3D Gabor wavelet函数。

    参数:
    x, y, b: 空间和频谱域的坐标网格。
    f: 中心频率。
    theta, phi: 频率向量的角度。
    sigma_x, sigma_y, sigma_b: 高斯包络在不同轴上的宽度。

    返回:
    3D Gabor wavelet的值。
    """
    # 旋转坐标
    x_theta = x * np.cos(theta) + y * np.sin(theta)
    y_theta = -x * np.sin(theta) + y * np.cos(theta)

    # 频谱域的旋转和缩放
    gb = np.exp(-.5 * (x_theta ** 2 + y_theta ** 2) / sigma_x ** 2) * \
         np.exp(-.5 * (b ** 2) / sigma_b ** 2) * \
         np.cos(2 * np.pi * f * x_theta + phi)

    return gb


def apply_gabor_3d(image, f, theta, phi, sigma_x, sigma_y, sigma_b):
    """
    将3D Gabor wavelet应用于高光谱图像。

    参数:
    image: 高光谱图像，维度为(height, width, bands)。
    f, theta, phi, sigma_x, sigma_y, sigma_b: 3D Gabor wavelet的参数。

    返回:
    卷积后的特征图。
    """
    # 创建坐标网格
    height, width, bands = image.shape
    x = np.linspace(-(width // 2), width // 2, width)
    y = np.linspace(-(height // 2), height // 2, height)
    b = np.linspace(-(bands // 2), bands // 2, bands)
    x, y, b = np.meshgrid(x, y, b)

    # 生成3D Gabor wavelet
    gabor_filter = gabor_3d(x, y, b, f, theta, phi, sigma_x, sigma_y, sigma_b)

    # 应用卷积
    feature_map = convolve(image, gabor_filter, mode='wrap')

    return feature_map


class FieldParser:
    def __init__(
            self,
            args
    ):
        super().__init__()
        self.args = args
        self.dataset = args.dataset
        self.vit_feature_extractor = AutoImageProcessor.from_pretrained(args.vision_model)
        self.i = 0
    def _parse_image(self, img):
        pixel_values = self.vit_feature_extractor(img, return_tensors="pt").pixel_values
        return pixel_values[0] 

    # from https://github.com/cuhksz-nlp/R2Gen/blob/main/modules/tokenizers.py
    def clean_report(self, report):
        # clean Iu-xray reports
        if self.dataset == "iu_xray":
            report_cleaner = lambda t: t.replace('..', '.').replace('..', '.').replace('..', '.').replace('1. ', '') \
            .replace('. 2. ', '. ').replace('. 3. ', '. ').replace('. 4. ', '. ').replace('. 5. ', '. ') \
            .replace(' 2. ', '. ').replace(' 3. ', '. ').replace(' 4. ', '. ').replace(' 5. ', '. ') \
            .strip().lower().split('. ')
            sent_cleaner = lambda t: re.sub('[.,?;*!%^&_+():-\[\]{}]', '', t.replace('"', '').replace('/', '').
                                            replace('\\', '').replace("'", '').strip().lower())
            tokens = [sent_cleaner(sent) for sent in report_cleaner(report) if sent_cleaner(sent) != []]
            report = ' . '.join(tokens) + ' .'
        # clean MIMIC-CXR reports
        else:
            report_cleaner = lambda t: t.replace('\n', ' ').replace('__', '_').replace('__', '_').replace('__', '_') \
                .replace('__', '_').replace('__', '_').replace('__', '_').replace('__', '_').replace('  ', ' ') \
                .replace('  ', ' ').replace('  ', ' ').replace('  ', ' ').replace('  ', ' ').replace('  ', ' ') \
                .replace('..', '.').replace('..', '.').replace('..', '.').replace('..', '.').replace('..', '.') \
                .replace('..', '.').replace('..', '.').replace('..', '.').replace('1. ', '').replace('. 2. ', '. ') \
                .replace('. 3. ', '. ').replace('. 4. ', '. ').replace('. 5. ', '. ').replace(' 2. ', '. ') \
                .replace(' 3. ', '. ').replace(' 4. ', '. ').replace(' 5. ', '. ').replace(':', ' :') \
                .strip().lower().split('. ')
            sent_cleaner = lambda t: re.sub('[.,?;*!%^&_+()\[\]{}]', '', t.replace('"', '').replace('/', '')
                                .replace('\\', '').replace("'", '').strip().lower())
            tokens = [sent_cleaner(sent) for sent in report_cleaner(report) if sent_cleaner(sent) != []]
            report = ' . '.join(tokens) + ' .' 
        # report = ' '.join(report.split()[:self.args.max_txt_len])

        return report

    def parse(self, features):
        to_return = {'id': features['id']}
        report = features.get("report", "")
        report = self.clean_report(report)
        to_return['input_text'] = report
        # if self.i <= 10:
        #     with open("/home/zhengweidong/projects/R2GenGPT/save/output.txt", "a") as file:
        #         k = features['id']
        #         file.write(f'{k}: '+f'{self.i}: ' + to_return['input_text'] + '\n')
        #         self.i = self.i + 1
        # chest x-ray images
        images = []
        j = 0
        for image_path in features['image_path']:
            k = features['id']

            with Image.open(os.path.join(self.args.base_dir, image_path)) as pil:
                array = np.array(pil, dtype=np.uint8)
                # if (self.i < 12):
                #     print(f'{k}: {self.i}' )
                #     from torchvision.transforms import ToPILImage
                #     to_pil = ToPILImage()
                #     pil_image = to_pil(array)
                #     pil_image.save(f"/home/zhengweidong/projects/R2GenGPT/save/init_output{self.i}_{j+1}_{k}.png")
                #     j = j + 1

                # # Gabor滤波
                # from torchvision.transforms import ToPILImage, ToTensor
                # to_pil = ToPILImage()
                # pil_image = to_pil(array)
                #
                # # Gabor滤波器参数
                # sigma_x = 1.0
                # sigma_y = 1.0
                # theta = np.pi / 4
                # lambd = 10.0
                # psi = 0.0
                # gamma = 0.2
                #
                # # 应用Gabor滤波器
                # image = apply_gabor_filter(array, sigma_x, sigma_y, theta, lambd, psi, gamma)
                # to_pil = ToPILImage()
                # image = to_pil(image)
                # image.save('/home/zhengweidong/projects/R2GenGPT/data/1.png')
                # exit()
                # # 转回tensor
                # to_tensor =  ToTensor()
                # array = to_tensor(image)

                #print("hhhh")
                if array.shape[-1] != 3 or len(array.shape) != 3:
                    array = np.array(pil.convert("RGB"), dtype=np.uint8)
                    # Gabor滤波
                    # 3D Gabor wavelet的参数
                f = 0.1  # 中心频率
                theta = np.pi / 4  # 频率向量的角度
                phi = np.pi / 2  # 频率向量的角度
                sigma_x = 3  # 高斯包络在x轴上的宽度
                sigma_y = 3  # 高斯包络在y轴上的宽度
                sigma_b = 1  # 高斯包络在频谱轴上的宽度

                # 应用3D Gabor wavelet
                # array = apply_gabor_3d(array, f, theta, phi, sigma_x, sigma_y, sigma_b)
                # from torchvision.transforms import ToPILImage
                # to_pil = ToPILImage()
                # image = to_pil(image)
                # print("save")
                # image.save('/home/zhengweidong/projects/R2GenGPT/data/2.png')
                # exit()
                image = self._parse_image(array)
                images.append(image)

                # if(self.i<12):
                #
                #     from torchvision.transforms import ToPILImage
                #     to_pil = ToPILImage()
                #     pil_image = to_pil(image)
                #     pil_image.save(f"/home/zhengweidong/projects/R2GenGPT/save/output{self.i}_{j}_{k}.png")

        to_return["image"] = images
        return to_return


    def transform_with_parse(self, inputs):
        return self.parse(inputs)


class ParseDataset(data.Dataset):
    def __init__(self, args, split='train'):
        self.args = args
        self.meta = json.load(open(args.annotation, 'r'))
        self.meta = self.meta[split]
        self.parser = FieldParser(args)

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, index):
        return self.parser.transform_with_parse(self.meta[index])


def create_datasets(args):
    train_dataset = ParseDataset(args, 'train')
    dev_dataset = ParseDataset(args, 'val')
    test_dataset = ParseDataset(args, 'test')
    return train_dataset, dev_dataset, test_dataset


