import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
import torchvision
from torchvision import datasets, transforms
import os
import numpy as np
import random
import torch.multiprocessing as mp
from datetime import datetime
from multiprocessing import Pool
from torch.cuda.amp import autocast#, GradScaler
from typing import Tuple

class VGG16(nn.Module):
    def __init__(self, num_classes=1000):
        super(VGG16, self).__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, num_classes)
        )
        
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

'''''
@torch.jit.script
def push_to_cuda(inputs, labels):
    return inputs.cuda(), labels.cuda()
'''''

def set_seeds(seed=0, all_deterministic=True):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    if all_deterministic:
        print("Using all deterministic!\n")
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True)
    g = torch.Generator()
    g.manual_seed(0)
    return g

def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)

def save_model(model, optimizer, epoch, lr, t_loss, vloss, top1_acc, top5_acc, timestamp, prefix='VGG16'):
    """Save the model to a file."""
    filename = f'{prefix}_epoch-{epoch}_lr-{lr:.1e}_vloss-{vloss:.4f}_top1-{top1_acc:.4f}_top5-{top5_acc:.4f}_{timestamp}.pth'
    filepath = os.path.join('saved_models', filename)
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': t_loss,
        'val_loss': vloss,
        'top1_acc': top1_acc,
        'top5_acc': top5_acc,
    }, filepath)


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def train_model(model, criterion, optimizer, scheduler, train_loader, val_loader, num_epochs=74):
    num_epochs = int(num_epochs)
    for epoch in range(num_epochs):
        
        
        print('\nEPOCH:', epoch+1, ' || Start Time:', datetime.now())
        model.train(True)
        
        running_loss = 0.0
        i = 0
        #with Pool(processes=4) as pool:
        for inputs, labels in train_loader:

            #stream_for_data_loading = torch.cuda.Stream()
            #stream_for_outputs = torch.cuda.Stream()
            #stream_for_loss = torch.cuda.Stream()

            #with torch.cuda.stream(stream_for_data_loading):
            #with Pool(processes=4) as pool:
            inputs, labels = inputs.cuda(non_blocking=True), labels.cuda(non_blocking=True)
                
            #with torch.cuda.stream(stream_for_outputs):
            optimizer.zero_grad(set_to_none=True)

            with autocast():
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            #with torch.cuda.stream(stream_for_loss):
            #scaler.scale(loss).backward()
            #scaler.step(optimizer)
            #scaler.update()

            #stream_for_data_loading.synchronize()
            #stream_for_outputs.synchronize()
            #stream_for_loss.synchronize()

            if i % 1000 == 999:
                last_loss = running_loss / 1000
                running_loss = 0.
                print('     batch {} || loss: {} || Timestamp: {}'.format(i + 1, last_loss, datetime.now()))
            i += 1
        epoch_loss = running_loss / len(train_loader.dataset)
        val_loss = evaluate_model(model, val_loader, criterion, optimizer, epoch_loss, epoch, num_epochs)
        scheduler.step(val_loss)
            

def evaluate_model(model, val_loader, criterion, optimizer, epoch_loss, epoch, num_epochs):
    model.eval()
    
    running_loss = 0.0
    top1_acc = 0.0
    top5_acc = 0.0
    
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.cuda(), labels.cuda()
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * inputs.size(0)
             
            acc1, acc5 = accuracy(outputs, labels, topk=(1, 5))
            top1_acc += acc1[0]
            top5_acc += acc5[0]
    
    val_loss = running_loss / len(val_loader.dataset)
    epoch_top1_acc = top1_acc / len(val_loader)
    epoch_top5_acc = top5_acc / len(val_loader)
    lr = optimizer.param_groups[0]['lr']
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_model(model, optimizer, epoch, lr, epoch_loss, val_loss, epoch_top1_acc, epoch_top5_acc, timestamp)
    print(f'{timestamp} -- Epoch {epoch}/{num_epochs - 1}, \n  Validation Loss: {epoch_loss:.4f}, \n   Top-1 Acc: {epoch_top1_acc:.4f}, Top-5 Acc: {epoch_top5_acc:.4f}')
    return val_loss


def main(rank, world_size, num_epochs, args):
    g = set_seeds(seed=0, all_deterministic=True)  # Set seeds for determinism

    transform = transforms.Compose([
        torchvision.transforms.Resize((256,256)),
        torchvision.transforms.RandomCrop((224,224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = datasets.ImageFolder(root= args.data_dir + 'train', transform=transform)
    val_dataset = datasets.ImageFolder(root= args.data_dir + 'sortedVal', transform= transform)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers = 14, worker_init_fn=worker_init_fn, generator=g)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers = 14, worker_init_fn=worker_init_fn, generator=g)

    model = torch.jit.script(VGG16())
    model = nn.DataParallel(model)
    model = model.cuda()

    #scaler = GradScaler()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=2)

    os.makedirs('saved_models', exist_ok=True)

    train_model(model, criterion, optimizer, scheduler, train_loader, val_loader, num_epochs)

if __name__ == '__main__':
    
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Train VGG16 Network"
    )
    parser.add_argument(
        "--data_dir",
        "-d",
        type=str,
        help="directory of the images data",
        default="/data/ImageNet/",
    )
    parser.add_argument(
        "--batch_size",
        "-b",
        type=int,
        default=128,
        help="batch size for the images passing through networks",
    )
    parser.add_argument(
        "--seed",
        "-s",
        type=int,
        help="momentum for training",
        default=0
    )
    parser.add_argument(
        "--num_epochs",
        "-ne",
        type=int,
        help="number of epochs to run",
        default=74
    )
    args = parser.parse_args()
    
    world_size = 1
    num_epochs = args.num_epochs
    rank = 'none'
    '''''
    mp.set_start_method('spawn')
    mp.spawn(main, args=(world_size, num_epochs, args,), nprocs=world_size, join=True)
    '''''

    '''''
    process = multiprocessing.Process(target=main,args=(rank, world_size, num_epochs, args,))
    process.start()
    process.join()
    '''''
    '''''
    with Pool(processes=2) as pool:
        pool.apply(main, args=(rank, world_size, num_epochs, args,))
    '''''
    mp.set_start_method('spawn')
    #main(rank, world_size, num_epochs, args)
    print('\nUsing mp.spawn!')
    mp.spawn(main, args=(world_size, num_epochs, args,), nprocs=world_size, join=True)

# take notes on what works/doesn't work, give reasons why as well
# each job could be a batch rather than a single image

# 4 processes was incredibly slow, way too much overhead, very low gpu utilization
# 2 processes a little better but still awful
# using 2 cuda streams alone (data loading and the rest) also slow
# using jit lowered to 8 min
