import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from PIL import Image
from torch.utils.data import DataLoader
import torchvision
from sklearn.metrics import confusion_matrix
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
if __name__ == "__main__":
    def get_data():
        train_transform = transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor(),transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
        train_data = torchvision.datasets.ImageFolder(root="Datasets/train",transform=train_transform)
        train_loader = torch.utils.data.DataLoader(train_data,batch_size=32,num_workers=8,shuffle=True,pin_memory=True)
        
        return train_loader,test_loader,test_data,train_data
    train_loader,test_loader,test_data,train_data = get_data()
    class Recycle(nn.Module):
        def __init__(self):
            super(Recycle,self).__init__()
            self.shortcut1 = nn.Sequential(nn.Conv2d(in_channels=3,out_channels=64,kernel_size=1,stride=1),nn.BatchNorm2d(64))
            self.block1 = nn.Sequential(nn.Conv2d(in_channels=3,out_channels=64,kernel_size=3,padding=1,stride=1),nn.BatchNorm2d(64),nn.ReLU(),nn.Conv2d(in_channels=64,out_channels=64,kernel_size=3,padding=1,stride=1),nn.BatchNorm2d(64))
            
            self.shortcut2 = nn.Sequential(nn.Conv2d(in_channels=64,out_channels=128,kernel_size=1,stride=2),nn.BatchNorm2d(128))
            self.block2 = nn.Sequential(nn.Conv2d(in_channels=64,out_channels=128,kernel_size=3,padding=1,stride=2),nn.BatchNorm2d(128),nn.ReLU(),nn.Conv2d(in_channels=128,out_channels=128,kernel_size=3,padding=1,stride=1),nn.BatchNorm2d(128))
            
            self.shortcut3 = nn.Sequential(nn.Conv2d(in_channels=128,out_channels=256,kernel_size=1,stride=2),nn.BatchNorm2d(256))
            self.block3 = nn.Sequential(nn.Conv2d(in_channels=128,out_channels=256,kernel_size=3,padding=1,stride=2),nn.BatchNorm2d(256),nn.ReLU(),nn.Conv2d(in_channels=256,out_channels=256,kernel_size=3,padding=1,stride=1),nn.BatchNorm2d(256))
           
            self.shortcut4 = nn.Sequential(nn.Conv2d(in_channels=256,out_channels=512,kernel_size=1,stride=2),nn.BatchNorm2d(512))
            self.block4 = nn.Sequential(nn.Conv2d(in_channels=256,out_channels=512,kernel_size=3,padding=1,stride=2),nn.BatchNorm2d(512),nn.ReLU(),nn.Conv2d(in_channels=512,out_channels=512,kernel_size=3,padding=1,stride=1),nn.BatchNorm2d(512))
            
            self.shortcut5 = nn.Sequential(nn.Conv2d(in_channels=512,out_channels=512,kernel_size=1,stride=1),nn.BatchNorm2d(512))
            self.block5 = nn.Sequential(nn.Conv2d(in_channels=512,out_channels=512,kernel_size=3,padding=1,stride=1),nn.BatchNorm2d(512),nn.ReLU(),nn.Conv2d(in_channels=512,out_channels=512,kernel_size=3,padding=1,stride=1),nn.BatchNorm2d(512))
            self.drop = nn.Dropout(0.3)
            self.relu = nn.ReLU()
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.classification = nn.Sequential(nn.Flatten(),nn.Linear(512,256),nn.ReLU(),nn.Linear(256,64),nn.ReLU(),nn.Linear(64,6))
        def forward(self,x):
                
            identity = self.shortcut1(x)
            out = self.block1(x)
            out = self.relu(out+ identity)

            identity = self.shortcut2(out)
            out = self.block2(out)
            out = self.relu(out+ identity)

            identity = self.shortcut3(out)
            out = self.block3(out)
            out = self.relu(out + identity)

            identity = self.shortcut4(out)
            out = self.block4(out)
            out = self.relu(out+identity)
            
            identity = self.shortcut5(out)
            out = self.block5(out)
            out = self.relu(out+identity)
            out = self.pool(out)

            out = self.classification(out)
            return out
    model = Recycle().to(device)
    model.load_state_dict(torch.load("Recycle2.pth",weights_only=False))
    optimizer = optim.Adam(model.parameters(),lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer,step_size=10,gamma=0.5)
    loss_calculate = nn.CrossEntropyLoss()
    def train(model,train_loader,optimizer,loss_calculate,device,epochs=1):
        model.train()
        for epoch in range(epochs):
            total_loss = 0
            for image,label in train_loader:
                image,label = image.to(device),label.to(device)
                guess = model(image)
                loss = loss_calculate(guess,label)
                optimizer.zero_grad()
                total_loss+=loss.item()
                loss.backward()
                optimizer.step()
            scheduler.step()
            print(f"Epoch: {epoch+1} Loss {total_loss/len(train_loader)}")    
        print("MOdel Eğitildi")
    
    train(model,train_loader,optimizer,loss_calculate,device,epochs=30)
    torch.save(model.state_dict(),"Recycle2.pth")    
