from torch.utils.data import DataLoader
from dataset import MyDataset
import torch
from torch import nn
from torch.nn import functional
from torch.optim import Adam
from net import Model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Train(object):
    Epoch = 10

    def __init__(self):
        super(Train, self).__init__()

    @staticmethod
    def train(model, optimizer, loss_func, train_dataloader):
        for epoch in range(Train.Epoch):
            sum_loss = 0
            for i, (data, label) in enumerate(train_dataloader):
                model.train()
                data = data.to(device)
                label = label.to(device)

                predict = model(data)
                loss = loss_func(predict, label.float())
                sum_loss += loss.item()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
            print(f'轮次:{epoch}\t损失:{sum_loss}')
        torch.save(model, 'model.pth')


class Predict:
    @staticmethod
    def predict():
        correct = 0
        total = 0
        model = torch.load('model.pth').to(device)
        test_dataset = MyDataset('Dataset/TestData')
        test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=True)
        for i, (image, label) in enumerate(test_dataloader):
            image = image.to(device)
            label = label.to(device)
            code = MyDataset.decode(label)
            predict = model(image)
            pre_code = MyDataset.decode(predict)
            if pre_code == code:
                correct += 1
            total += 1
        print(f'正确率:{correct / total:.4f}')


if __name__ == '__main__':
    MODEL = Model().to(device)
    loss_function = nn.MSELoss().to(device)
    opt = Adam(MODEL.parameters(), lr=0.001)
    dataloader = DataLoader(MyDataset('Dataset/train_img'), batch_size=64, shuffle=True)
    Train.train(MODEL, opt, loss_function, dataloader)
    Predict.predict()

