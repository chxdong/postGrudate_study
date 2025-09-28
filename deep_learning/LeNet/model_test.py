import torch
import torch.utils.data as Data
from torchvision import transforms
from torchvision.datasets import FashionMNIST
from model import LeNet

def test_val_data_process():
    test_data = FashionMNIST(root='./data',
                              train=False,
                              transform=transforms.Compose([transforms.Resize(size=28), transforms.ToTensor()]),
                              download=True
                              )

    test_dataloader = Data.DataLoader(dataset=test_data,
                                       batch_size=1,
                                       shuffle=True,
                                       num_workers=2)

    return test_dataloader

def test_model_process(model, test_dataloader):
    device = "cuda" if torch.cuda.is_available() else "CPU"

    model = model.to(device)

    test_corrects = 0.0
    test_num = 0

    with torch.no_grad():
        for test_data_x,test_data_y in test_dataloader:
            test_data_x = test_data_x.to(device)
            test_data_y = test_data_y.to(device)
            model.eval()
            output = model(test_data_x)
            pre_lab = torch.argmax(output, dim =1)

            test_corrects += torch.sum(pre_lab == test_data_y.data)



