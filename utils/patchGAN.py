import torch.nn as nn

class FullResDiscriminatorText(nn.Module):
    def __init__(self, in_channels=4, ndf=64, text_dim=512):
        super().__init__()


        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, ndf, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(ndf, ndf*2, 4, 2, 1),
            nn.BatchNorm2d(ndf*2),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.enc3 = nn.Sequential(
            nn.Conv2d(ndf*2, ndf*4, 4, 2, 1),
            nn.BatchNorm2d(ndf*4),
            nn.LeakyReLU(0.2, inplace=True)
        )


        self.text_proj = nn.Linear(text_dim, ndf*4)

        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(ndf*4, ndf*2, 4, 2, 1),
            nn.BatchNorm2d(ndf*2),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(ndf*2, ndf, 4, 2, 1),
            nn.BatchNorm2d(ndf),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.dec3 = nn.Sequential(
            nn.ConvTranspose2d(ndf, 1, 4, 2, 1),
            nn.Sigmoid()
        )

    def forward_features(self, x, text_feat):
        f1 = self.enc1(x)
        f2 = self.enc2(f1)
        f3 = self.enc3(f2)

        text_cond = self.text_proj(text_feat).unsqueeze(-1).unsqueeze(-1)
        f3 = f3 + text_cond

        return f1, f2, f3

    def forward(self, x, text_feat):
        f1, f2, f3 = self.forward_features(x, text_feat)
        y = self.dec1(f3)
        y = self.dec2(y)
        y = self.dec3(y)
        return y
