import torch
import torchvision
from torchvision import datasets
import torch.nn as nn

def accuracyk(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

class VGG16(nn.Module):
    def __init__(self, num_classes=1000):
        super(VGG16, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU())
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(), 
            nn.MaxPool2d(kernel_size = 2, stride = 2))
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU())
        self.layer4 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2))
        self.layer5 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU())
        self.layer6 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU())
        self.layer7 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2))
        self.layer8 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU())
        self.layer9 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU())
        self.layer10 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2))
        self.layer11 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU())
        self.layer12 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU())
        self.layer13 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2))
        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(7*7*512, 4096),
            nn.ReLU())
        self.fc1 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU())
        self.fc2= nn.Sequential(
            nn.Linear(4096, num_classes))
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out)
        out = self.layer7(out)
        out = self.layer8(out)
        out = self.layer9(out)
        out = self.layer10(out)
        out = self.layer11(out)
        out = self.layer12(out)
        out = self.layer13(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        out = self.fc1(out)
        out = self.fc2(out)
        return out

if __name__ == "__main__":


    transforms_testing = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),  
        torchvision.transforms.Resize((256,256)),
        torchvision.transforms.CenterCrop((224,224))
    ])

    test_dataset = datasets.ImageFolder(root= "/data/ImageNet/" + 'sortedVal', transform= transforms_testing)

    # Creating Data Loaders
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers = 14)

    # Choosing device as GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # initializing epoch number
    epoch_number = 0

    # initializing VGG16 model
    model = VGG16()
    # using GPUs
    model = torch.nn.DataParallel(model, device_ids=[0,1])
    model = model.to(device)

    torch.autocast('cpu', enabled=False)

    loss_fn = torch.nn.CrossEntropyLoss()
    loss_fn = loss_fn.to(device)

    print('Using Pretrained Model State')
    pretrain_name = '/data/brustdm/model_imp/VGG16ModelStates/model_ts-2023-07-30 19:07:24.981884_ep-49_et-10707.804024457932_tt--1 day, 20:58:46.164602_st-44.49707102775574_bv-True_ac-0.65996'
    model.load_state_dict(torch.load(pretrain_name))

    if 1 == 1:

        running_vloss = 0.0
        top1_correct_total = 0.0
        top5_correct_total = 0.0
        total_samples = 0.0
        # Set the model to evaluation mode, disabling dropout and using population
        # statistics for batch normalization.
        # want to know about model.eval()
        # diabling dropout? == Yes
        model.eval()

        # Disable gradient computation and reduce memory consumption.
        with torch.no_grad():
            for i, vdata in enumerate(test_loader):
                vinputs, vlabels = vdata
                vinputs, vlabels = vinputs.to(device), vlabels.to(device)
                voutputs = model(vinputs)
                vloss = loss_fn(voutputs, vlabels)
                running_vloss += vloss
                top1_correct_total += (voutputs.argmax(1) == vlabels).sum().item()
                top5 = torch.topk(voutputs, 5)
                for top in top5:
                    top5_correct_total += (voutputs.argmax(1) == vlabels).sum().item()
                total_samples += vlabels.size(0)
        avg_vloss = running_vloss / (i + 1)
        accuracy1 = top1_correct_total / total_samples
        accuracy5 = top5_correct_total / total_samples
        
        accs = accuracyk(voutputs, vlabels, topk=(1,5))
        
        print('LOSS: valid {}'.format(avg_vloss))
        print('Top1 Accuracy:', accuracy1)
        print('Top5 Accuracy:', accuracy1)
        print('Top1 Accuracy k:', accs[0])
        print('Top5 Accuracy k:', accs[1])
        print('accs:', accs)