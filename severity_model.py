
import torch
import torch.nn as nn
import torchvision.models as models

NUM_DISEASES  = 13
NUM_SEVERITY  = 4
FEATURE_SIZE  = 1024
COMBINED_SIZE = FEATURE_SIZE + NUM_DISEASES  

SEV_LABELS = ['N/A', 'mild', 'moderate', 'severe']


class SeverityModel(nn.Module):
   

    def __init__(self, pretrained: bool = True):
        super(SeverityModel, self).__init__()

        if pretrained:
            backbone = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
        else:
            backbone = models.densenet121(weights=None)

        self.features = backbone.features
        self.gap      = nn.AdaptiveAvgPool2d((1, 1))

        self.head = nn.Sequential(
            nn.Linear(COMBINED_SIZE, 256),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(256, NUM_SEVERITY),
        )

    def forward(self, x: torch.Tensor, disease_idx: torch.Tensor) -> torch.Tensor:
       
        features = self.features(x)
        features = torch.relu(features)
        pooled   = self.gap(features)
        pooled   = pooled.view(pooled.size(0), -1)     

        
        one_hot = torch.zeros(
            pooled.size(0), NUM_DISEASES,
            device=x.device, dtype=torch.float32
        )
        one_hot.scatter_(1, disease_idx.unsqueeze(1), 1.0)  

    
        combined = torch.cat([pooled, one_hot], dim=1)      

        
        return self.head(combined)                          

    @torch.no_grad()
    def predict_proba(self, x: torch.Tensor, disease_idx: torch.Tensor) -> torch.Tensor:
      
        logits = self.forward(x, disease_idx)
        return torch.softmax(logits, dim=1)


def load_severity_model(weights_path: str, device: torch.device) -> SeverityModel:
  
    model = SeverityModel(pretrained=False)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device)
    model.eval()
    return model