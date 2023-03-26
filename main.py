import torch
from classes import classes
import torchvision
from torch import nn
from PIL import Image

device = 'cuda' if torch.cuda.is_available() else 'cpu'
image_path = input('enter_image_path:')

weights = torchvision.models.EfficientNet_B0_Weights.DEFAULT 
transforms = weights.transforms()
model = torchvision.models.efficientnet_b0(weights=weights).to(device)

model.classifier = torch.nn.Sequential(
    torch.nn.Dropout(p=0.2, inplace=True), 
    torch.nn.Linear(in_features=1280, 
                    out_features=510,
                    bias=True))

model.load_state_dict(torch.load('birds-510-state_dict.pt',map_location=torch.device('cpu')))


# Load the image and apply the transformations
image = Image.open(image_path)
image = transforms(image)

# Apply the model to the image
with torch.inference_mode():
    model.eval()
    output = classes[torch.softmax(model(image.unsqueeze(0)),dim=1).argmax(dim=1)]


print(output)
