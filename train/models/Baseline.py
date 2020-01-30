import torch
import torch.nn as nn

class Baseline(nn.Module):
    def __init__(self):
        super(Baseline,self).__init__()
        # input torch.Size([4, 1, 512, 512])
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=4, stride=2, padding=0),
            nn.ReLU(inplace=True),
            nn.AdaptiveMaxPool2d((128,128))
        )  # torch.Size([4, 64, 128, 128])
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(inplace=True),
            nn.AdaptiveMaxPool2d((32,32))
        )  # torch.Size([4, 64, 32, 32])
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=2, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveMaxPool2d((8,8))
        )  # torch.Size([4, 128, 8, 8])
        

        self.fc1 = nn.Sequential(
            nn.Linear(8192,1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024,20)
            # nn.Sigmoid()
        )

    def forward(self, x):
        x = self.conv1(x)
        #print(x.size())
        x = self.conv2(x)
        #print(x.size())
        x = self.conv3(x)
        #print(x.size())
        x = x.view(x.size(0),-1)
        x = self.fc1(x)
        return x

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.xavier_uniform(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()