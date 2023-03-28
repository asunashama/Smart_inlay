import re
import torch
from torch.utils.data import Dataset
import os
from PIL import Image
from torchvision import transforms



class MyDataset(Dataset):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((40, 100)),
        transforms.Grayscale()
    ])

    def __init__(self, image_path):
        image_path = 'Verification_code_identification\\source\\train_img'
        super(MyDataset, self).__init__()
        self.image_path = [os.path.join(image_path, file_name) for file_name in os.listdir(image_path)]

    def __len__(self):
        return len(self.image_path)

    def __getitem__(self, item):
        image = Image.open(self.image_path[item])
        image = MyDataset.transform(image)
        image_name = self.image_path[item].split('/')[-1]
        code = image_name.split('_')[0]
        code = MyDataset.encode(code)  # 转换为展平的单热点编码tensor
        return image, code

    @staticmethod
    def encode(code):
        """
        :param code:验证码
        :return: 展平后的单热点编码Tensor
        使用单热点编码，便于神经网络输出
        """
        all_code = list('0123456789')
        code = ''.join(re.findall(r'\d', code))
        encoded = torch.zeros(len(code), len(all_code), dtype=torch.int)
        for i in range(len(code)):
            encoded[i, all_code.index(code[i])] = int(1)
        encoded = torch.flatten(encoded)
        return encoded

    @staticmethod
    def decode(code_tensor):
        all_code = list('0123456789')
        f = code_tensor.view(4, 10)
        result = []
        for row in f:
            result.append(all_code[torch.argmax(row, dim=0)])
        result = ''.join(result)
        return result

