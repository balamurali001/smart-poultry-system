import torch.nn as nn
import timm

class ChickenDiseaseDetector(nn.Module):
  """
  Model Arch replicates TinyVGG model
  from CNN explainer website

  """

  def __init__(self, input_shape:int, hidden_units:int, output_shape:int,image_dimension:int):
    super().__init__()
    self.conv_block_1=nn.Sequential(
        nn.Conv2d(in_channels=input_shape,
                  out_channels=hidden_units,
                  kernel_size=3,
                  stride=1,
                  padding=1),
        nn.ReLU(),
        nn.Conv2d(in_channels=hidden_units,
                  out_channels=hidden_units,
                  kernel_size=3,
                  stride=1,
                  padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2)
    )

    self.conv_block_2=nn.Sequential(
    nn.Conv2d(in_channels=hidden_units,
                  out_channels=hidden_units,
                  kernel_size=3,
                  stride=1,
                  padding=1),
    nn.ReLU(),
    nn.Conv2d(in_channels=hidden_units,
                  out_channels=hidden_units,
                  kernel_size=3,
                  stride=1,
                  padding=1),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2)
    )

    self.classifier=nn.Sequential(
        nn.Flatten(),
        nn.Linear(in_features=hidden_units*image_dimension//2//2*image_dimension//2//2,
                  out_features=output_shape)
    )

  def forward(self,x):
    return self.classifier(self.conv_block_2(self.conv_block_1(x)))


class MobileeNetV2(nn.Module):
    def __init__(self, num_classes: int):
        super(MobileeNetV2, self).__init__()
        
        # Load EfficientNetB3 base model
        self.base_model = timm.create_model(
            'mobilenetv2_100', 
            pretrained=True,
            num_classes=0  # Remove original classification head
        )
        
        # Freeze base model if fine-tuning is not needed
        # for param in self.base_model.parameters():
        #     param.requires_grad = False
        
        # Add custom classification head
        self.classifier = nn.Sequential(
            nn.Linear(self.base_model.num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.base_model(x)
        x = self.classifier(x)
        return x
