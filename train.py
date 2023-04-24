import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import cv2
import os
from torchvision.transforms import transforms
from tqdm import tqdm
from models.unet import UNet
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt

from models.SegFormer import Segformer
from models.unetplusplus.unetplusplus import NestedUNet

parser = argparse.ArgumentParser()

parser.add_argument('--model', metavar='',type=str, default='unet', help='The architecture used, which can be unet, SegFormer and unet++')
parser.add_argument('--epochs',metavar='', type=int, default=100, help='Number of epochs')
parser.add_argument('--batch_size',metavar='', type=int, default=1, help='Batch size')
parser.add_argument('--learning_rate',metavar='', type=float, default=1e-5, help='Learning rate')
parser.add_argument('--path_to_images',metavar='',type=str,default='dataset/train/Images',help='path to images')
parser.add_argument('--path_to_masks',metavar='',type=str,default='dataset/train/Segmentation1',help='path to masks')
parser.add_argument('--scale',metavar='', type=float, default=0.3, help='Downscaling factor of the images')

args = parser.parse_args()


# define the training dataset
class MandibleDataset(Dataset):
    def __init__(self, path_to_images,path_to_masks, transform=None):
        self.path_to_images =path_to_images
        self.path_to_masks= path_to_masks
        self.transform = transform
        self.images = []
        self.masks = []
        
        for filename in os.listdir(self.path_to_images):
            img = cv2.imread(os.path.join(self.path_to_images, filename), 0)
            if args.model=='unet':
                img=cv2.resize(img , ( int(img.shape[1]*args.scale),int(img.shape[0]*args.scale) ))
            elif args.model=='SegFormer':
                img=cv2.resize(img , (1024, 512))
            else:
                img=cv2.resize(img , (512, 512))

            img-=img.min()
            img = img.astype(np.float32) / img.max()
            self.images.append(img)
        i=-1
        for filename in os.listdir(self.path_to_masks):
            i+=1
            masks = cv2.imread(os.path.join(self.path_to_masks, filename), 0)
            if args.model=='unet':
                masks=cv2.resize(masks , ( self.images[i].shape[1],self.images[i].shape[0] ))
            elif args.model=='SegFormer':
                masks=cv2.resize(masks , (256, 128))
            else:     
                masks=cv2.resize(masks , (512, 512))

            masks-=masks.min()
            masks = masks.astype(np.float32) / masks.max()
            masks=np.where(masks==np.unique(masks)[0],0,1)
            self.masks.append(masks)


    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        mask = self.masks[idx]
        if self.transform is not None:

            image = self.transform(image).to(device).contiguous()
            mask = self.transform(mask).to(device).contiguous()
        return image, mask


def Dice(output,target,weight=None, eps=1e-5):
    target = target.float()
    if weight is None:
        num = 2 * (output * target).sum()
        den = output.sum() + target.sum() + eps
    return 1.0 - num/den



# define the training function
def train(model, device, train_loader, optimizer, criterion):
    model.train()
    train_loss = 0
    total_dice = 0
    for batch_idx, (data, target) in enumerate(tqdm(train_loader)):
        data, target = data.to(device), target.to(device)
        target=target.squeeze(1).long()
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss+=Dice(output[:,1,:,:], target)
        train_loss += loss.item()
        loss.backward()
        optimizer.step()
        total_dice += Dice(output[:,0,:,:], target).item()
    train_loss /= len(train_loader)
    total_dice /= len(train_loader)
    return train_loss, total_dice

# define the validation function
def validate(model, device, val_loader, criterion):
    model.eval()
    val_loss = 0
    total_dice = 0
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            target=target.squeeze(1).long()
            output = model(data)
            loss = criterion(output, target)
            loss+=Dice(output[:,1,:,:], target)
            val_loss += loss.item()
            total_dice += Dice(output[:,0,:,:], target).item()
    val_loss /= len(val_loader)
    total_dice /= len(val_loader)
    return val_loss, total_dice



# define the function to plot the train and validation loss
def plot_loss(train_loss, val_loss):
    plt.plot(train_loss, label='Train loss')
    plt.plot(val_loss, label='Validation loss')
    plt.title('Train and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

# define the function to plot the train and validation dice score
def plot_dice(train_dice, val_dice):
    plt.plot(train_dice, label='Train dice score')
    plt.plot(val_dice, label='Validation dice score')
    plt.title('Train and Validation Dice Score')
    plt.xlabel('Epoch')
    plt.ylabel('Dice score')
    plt.legend()
    plt.show()




def train_model(model,epochs,batch_size,learning_rate,device):

    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    criterion =  nn.CrossEntropyLoss() 
    optimizer = optim.RMSprop(model.parameters(), lr=learning_rate)

    # define the training data loader
    dataset = MandibleDataset(args.path_to_images,args.path_to_masks, transform=transform)


    # Split into train / validation partitions
    n_val = int(len(dataset) * 0.2)
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(13))

    train_loader = DataLoader(train_set,batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=True)

    # define the lists to store the train and validation loss and dice score
    train_loss_list = []
    val_loss_list = []
    train_dice_list = []
    val_dice_list = []


    min_val_loss=1000
    # train the model
    for epoch in range(epochs):
        print('Epoch {}/{}'.format(epoch+1, epochs))
        train_loss, train_dice = train(model, device, train_loader, optimizer, criterion)
        print('Train loss: {:.6f}, Train dice score: {:.6f}'.format(train_loss, train_dice))
        val_loss, val_dice = validate(model, device, val_loader, criterion)
        print('Validation loss: {:.6f}, Validation dice score: {:.6f}'.format(val_loss, val_dice))
        train_loss_list.append(train_loss)
        val_loss_list.append(val_loss)
        train_dice_list.append(train_dice)
        val_dice_list.append(val_dice)
        torch.save(model.state_dict(), 'checkpoints/{}_model_epoch{}.pth'.format(args.model,epoch+1))
        if val_loss<min_val_loss:
            torch.save(model.state_dict(), 'checkpoints/{}_model_best.pth'.format(args.model))


    # plot the train and validation loss and dice score
    plot_loss(train_loss_list, val_loss_list)
    plot_dice(train_dice_list, val_dice_list)

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if args.model=='unet':
        model = UNet(n_channels=1, n_classes=2, bilinear=False)
    if args.model=='SegFormer':
        model=Segformer(channels=1,num_classes=2)
    if args.model=='unet++':
        model=NestedUNet(1,2)

    model = model.to(memory_format=torch.channels_last)
    model.to(device=device)

    if not os.path.exists('checkpoints'):
        os.mkdir('checkpoints')


    train_model(
        model=model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        device=device)