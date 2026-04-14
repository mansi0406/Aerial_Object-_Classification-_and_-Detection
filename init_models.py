import torch
import torchvision.models as models
import torch.nn as nn
import os

def init_models():
    # ResNet50
    resnet = models.resnet50(weights=None)
    num_features = resnet.fc.in_features
    resnet.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(num_features, 256),
        nn.ReLU(inplace=True),
        nn.Dropout(0.3),
        nn.Linear(256, 2)
    )
    torch.save(resnet.state_dict(), 'best_resnet50.pth')
    print("Initialized best_resnet50.pth (untrained)")

    # VGG16
    vgg = models.vgg16(weights=None)
    vgg.classifier[6] = nn.Linear(4096, 2)
    torch.save(vgg.state_dict(), 'best_vgg16.pth')
    print("Initialized best_vgg16.pth (untrained)")

    # Custom CNN
    from train import CustomCNN
    custom_cnn = CustomCNN(num_classes=2)
    torch.save(custom_cnn.state_dict(), 'best_custom_cnn.pth')
    print("Initialized best_custom_cnn.pth (untrained)")

if __name__ == '__main__':
    init_models()
