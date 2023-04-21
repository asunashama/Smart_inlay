import torch
import torchvision
import numpy as np
import cv2


# 定义相关常数
num_classes = 2  # 行人和背景两个类别
device = torch.device('cuda')  # 使用GPU加速推理
model_path = 'path/to/model.pth'  # 训练好的模型路径
# 定义Faster R-CNN模型
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False, num_classes=num_classes)

# 加载训练好的权重
model.load_state_dict(torch.load(model_path))
model.to(device)
model.eval()
def detect_pedestrians(image):
    # 将图像转换为tensor格式并标准化
    image_tensor = torchvision.transforms.functional.to_tensor(image)
    image_tensor = torchvision.transforms.functional.normalize(image_tensor, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    image_tensor = image_tensor.to(device)

    # 进行推理
    with torch.no_grad():
        outputs = model([image_tensor])

    # 后处理
    boxes = outputs[0]['boxes'].cpu().detach().numpy()
    scores = outputs[0]['scores'].cpu().detach().numpy()

    # 选择置信度较高的框
    pedestrian_boxes = boxes[scores >= 0.5]

    # 将检测结果绘制在图像上
    for box in pedestrian_boxes:
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), thickness=2)
        
    return image, len(pedestrian_boxes)
# 读取图像
image = cv2.imread('path/to/image.jpg')

# 对图像进行检测
result_image, num_pedestrians = detect_pedestrians(image)

# 显示结果
cv2.imshow('Result', result_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

print(f'Found {num_pedestrians} pedestrians.')
