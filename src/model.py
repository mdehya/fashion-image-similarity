import torch
import torchvision.models as models
import torchvision.transforms as transforms

def get_model(device):
    model = models.resnet50(pretrained=True)
    
    # Hapus fully connected layer
    model = torch.nn.Sequential(*list(model.children())[:-1])
    
    model.eval()
    model.to(device)
    
    return model


def get_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),  # resize dari 60x80
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
