from PIL import Image
import torch
from torch import nn
from torchvision import models, transforms

trained_model = None
class_names = ["Front Breakage", "Front Crushed", "Front Normal","Rear Breakage", "Rear Crushed", "Rear Normal"]
#Load the pre-trained ResNet50 model and modify the final layer
class CarClassifierResNet(nn.Module):
    def __init__(self, num_classes=6):
        super().__init__()
        self.model = models.resnet50(weights="DEFAULT")
        # Freeze all layers except final fully connected layer
        for param in self.model.parameters():
            param.requires_grad = False

        # Unfreezer layer4 and fc layers
        for param in self.model.layer4.parameters():
            param.requires_grad = True
        
        in_features = self.model.fc.in_features
        
        self.model.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(in_features, num_classes)
        )
    def forward(self, x):
        x = self.model(x)
        return x
    
def predict(image_path):
    image = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image_tensor = transform(image).unsqueeze(0) # (1,32,2,224,224)
    
    global trained_model
    if trained_model is None:
        trained_model = CarClassifierResNet()
        trained_model.load_state_dict(torch.load("model/saved_model.pth"))
        trained_model.eval()
    
    
    with torch.no_grad():
        outputs = trained_model(image_tensor) # [[12.3, 2.4, 5.6, 3.2, 1.2, 0.3]]
        _, predicted_class = torch.max(outputs, 1) # [1]
        return class_names[predicted_class.item()]