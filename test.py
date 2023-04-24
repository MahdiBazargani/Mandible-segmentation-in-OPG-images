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
parser.add_argument('--checkpoint', metavar='',type=str,default='checkpoints/unet_model_best.pth',help='path to model')
parser.add_argument('--path_to_images',metavar='',type=str,default='dataset/test/Images',help='path to images')
parser.add_argument('--path_to_masks',metavar='',type=str,default='dataset/test/Segmentation1',help='path to masks')
parser.add_argument('--output_dir',metavar='',type=str,default='dataset/test/outputs',help='path to output directory')
parser.add_argument('--scale',metavar='', type=float, default=0.3, help='Downscaling factor of the images')

args = parser.parse_args()



# define the test dataset
class MandibleDataset(Dataset):
    def __init__(self, path_to_images,path_to_masks, transform=None):
        self.path_to_images = path_to_images
        self.path_to_masks= path_to_masks
        self.transform = transform
        self.images = []
        self.masks = []
        self.sizes=[]
        self.filenames=[]
        self.main_images=[]
        for filename in os.listdir(self.path_to_images):
            img = cv2.imread(os.path.join(self.path_to_images, filename), 0)
            self.main_images.append(img)
            self.sizes.append(img.shape)

            if args.model=='unet':
                img=cv2.resize(img , ( int(img.shape[1]*args.scale),int(img.shape[0]*args.scale) ))
            elif args.model=='SegFormer':
                img=cv2.resize(img , (1024, 512))
            else:
                img=cv2.resize(img , (512, 512))

            img-=img.min()
            img = img.astype(np.float32) / img.max()            
            self.images.append(img)
            self.filenames.append(filename)
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
        size = self.sizes[idx]
        filename=self.filenames[idx]
        main_image=self.main_images[idx]

        if self.transform is not None:

            image = self.transform(image).to(device).contiguous()
            mask = self.transform(mask).to(device).contiguous()
        return image, mask, size, filename, main_image


def Dice(output,target,weight=None, eps=1e-5):
    target = target.float()
    if weight is None:
        num = 2 * (output * target).sum()
        den = output.sum() + target.sum() + eps
    return 1.0 - num/den


# define the test function
def test(model, device, val_loader, criterion):
    model.eval()
    val_loss = 0
    total_dice = 0
    with torch.no_grad():
        for data, target,size,filename,main_image in val_loader:
            data, target = data.to(device), target.to(device)
            target=target.squeeze(1).long()
            output = model(data)
            main_image=main_image.cpu().detach().numpy()
            mask=cv2.resize(output[0,0].cpu().detach().numpy(),(size[1].item(),size[0].item()))
            mask=np.where(mask>0.5,0,255)
            result=np.concatenate([main_image,main_image,main_image])
            result[2,:,:]=np.where(mask==255,200,result[2,:,:])
            result=np.transpose(result,(1,2,0))
            path=os.path.join(args.output_dir,filename[0])
            print(path)
            cv2.imwrite(path,result)
            loss = criterion(output, target)
            loss+=Dice(output[:,1,:,:], target)
            val_loss += loss.item()
            total_dice += Dice(output[:,0,:,:], target).item()
            print(Dice(output[:,0,:,:], target).item())

    val_loss /= len(val_loader)
    total_dice /= len(val_loader)
    return val_loss, total_dice



def test_model(model,device):
    # define the data transforms
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    # loading testset
    dataset = MandibleDataset(args.path_to_images,args.path_to_masks, transform=transform)
    test_loader = DataLoader(dataset, shuffle=False, drop_last=True)

    criterion =  nn.CrossEntropyLoss() 
    test_loss, test_dice = test(model, device, test_loader, criterion)
    print('test loss: {}'.format(test_loss))
    print('dice score: {}'.format(test_dice))


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if args.model=='unet':
        model = UNet(n_channels=1, n_classes=2, bilinear=False)
    if args.model=='SegFormer':
        model=Segformer(channels=1,num_classes=2)
    if args.model=='unet++':
        model=NestedUNet(1,2)
    model = model.to(memory_format=torch.channels_last)

    state_dict = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device=device)
    

    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    test_model(model=model,device=device)