import torch.nn as nn

class VGG_A(nn.Module):
    def __init__(self, inp_ch=3, num_classes=10, activation='relu', init_weights=True):
        super().__init__()
        if activation == 'relu':
            act_layer = nn.ReLU(True)
        elif activation == 'leaky_relu':
            act_layer = nn.LeakyReLU(0.1, True)
        elif activation == 'sigmoid':
            act_layer = nn.Sigmoid()
        else:
            raise ValueError("Unsupported activation function")

        self.features = nn.Sequential(
            nn.Conv2d(inp_ch, 64, 3, padding=1),
            act_layer,
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, padding=1),
            act_layer,
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, 3, padding=1),
            act_layer,
            nn.Conv2d(256, 256, 3, padding=1),
            act_layer,
            nn.MaxPool2d(2, 2),
            nn.Conv2d(256, 512, 3, padding=1),
            act_layer,
            nn.Conv2d(512, 512, 3, padding=1),
            act_layer,
            nn.MaxPool2d(2, 2),
            nn.Conv2d(512, 512, 3, padding=1),
            act_layer,
            nn.Conv2d(512, 512, 3, padding=1),
            act_layer,
            nn.MaxPool2d(2, 2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(512, 512),
            act_layer,
            nn.Linear(512, 512),
            act_layer,
            nn.Linear(512, num_classes)
        )
        if init_weights:
            self._init_weights()

    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, 512)
        x = self.classifier(x)
        return x

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

class VGG_A_BatchNorm(nn.Module):
    def __init__(self, inp_ch=3, num_classes=10, activation='relu', init_weights=True):
        super().__init__()
        if activation == 'relu':
            act_layer = nn.ReLU(True)
        elif activation == 'leaky_relu':
            act_layer = nn.LeakyReLU(0.1, True)
        elif activation == 'sigmoid':
            act_layer = nn.Sigmoid()
        else:
            raise ValueError("Unsupported activation function")
        self.features = nn.Sequential(
            nn.Conv2d(inp_ch, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            act_layer,
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            act_layer,
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            act_layer,
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            act_layer,
            nn.MaxPool2d(2, 2),
            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            act_layer,
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            act_layer,
            nn.MaxPool2d(2, 2),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            act_layer,
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            act_layer,
            nn.MaxPool2d(2, 2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            act_layer,
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            act_layer,
            nn.Linear(512, num_classes)
        )
        if init_weights:
            self._init_weights()

    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, 512)
        x = self.classifier(x)
        return x

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.constant_(m.bias, 0) 
