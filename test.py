import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from torch.utils.data import DataLoader
import torchvision
from torch import optim
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if __name__ == "__main__":
    def get_data():
        test_transform = transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        test_data = torchvision.datasets.ImageFolder(root="Datasets/test", transform=test_transform)
        test_loader = torch.utils.data.DataLoader(test_data, batch_size=32, shuffle=False)
        return test_data,test_loader
    test_data,test_loader = get_data()
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
    def test(model,test_loader,loss_calculate,device):
        model.eval()
        correct = 0
        total = 0
        test_loss = 0
        with torch.no_grad():
            for image, label in test_loader:
                    image, label = image.to(device), label.to(device)
                    outputs = model(image)
                    loss = loss_calculate(outputs, label)
                    test_loss += loss.item()
                    _, predicted = torch.max(outputs, 1)
                    total += label.size(0)
                    correct += (predicted == label).sum().item()
            print(f"Test Loss: {test_loss/len(test_loader):.4f}")
            print(f"Test Accuracy: {100 * correct / total:.2f}%")
    test(model,test_loader,loss_calculate,device)
