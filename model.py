

import torch
import torch.nn as nn
import torchvision.models as models

NUM_CLASSES       = 13   
NUM_SEVERITY      = 4     
DENSENET_FEATURES = 1024  


class XrayModel(nn.Module):
    
    def __init__(self, pretrained: bool = True):
        super(XrayModel, self).__init__()

        if pretrained:
            backbone = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
        else:
            backbone = models.densenet121(weights=None)


        self.features = backbone.features

  
        self.gap = nn.AdaptiveAvgPool2d((1, 1))

   
        self.prob_head = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(DENSENET_FEATURES, NUM_CLASSES),
            nn.Sigmoid(),
        )

      
        self.sev_head = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(DENSENET_FEATURES, NUM_CLASSES * NUM_SEVERITY),
        )

    def forward(self, x: torch.Tensor):
        
        features = self.features(x)
        features = torch.relu(features)
        pooled   = self.gap(features)
        pooled   = pooled.view(pooled.size(0), -1)

        prob_out = self.prob_head(pooled)

        sev_raw = self.sev_head(pooled)
        sev_raw = sev_raw.view(-1, NUM_CLASSES, NUM_SEVERITY)
        sev_out = torch.softmax(sev_raw, dim=2)

        return prob_out, sev_out


def load_model(weights_path: str, device: torch.device) -> XrayModel:

    model = XrayModel(pretrained=False)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device)
    model.eval()
    return model