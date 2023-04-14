from PIL import Image
import torchvision.transforms as T


def get_transforms(train):
    if train:
        transforms = T.Compose([
            T.Resize((224, 224)),
            T.RandomHorizontalFlip(),
            T.RandomApply([
                T.ColorJitter(
                    brightness=0.8, 
                    contrast=0.8, 
                    saturation=0.8,
                    hue=0.2), 
                T.RandomAffine(
                    18, 
                    scale=(0.9, 1.1),
                    translate=(0.1, 0.1), 
                    shear=10,
                    resample=Image.BILINEAR, 
                    fillcolor=0)
            ], p=0.5),
            T.RandomGrayscale(p=0.2),
            T.ToTensor(),
            T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
    else:
        transforms = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
        
    return transforms