import torch
import torchvision.transforms as T

def augment_resnet_images(X, train=True):
    """
    Apply standard ResNet18 data augmentation to image tensors.

    Args:
        X (torch.Tensor): Image tensor of shape [N, 3, H, W]
        train (bool): If True, apply data augmentation.
                      If False, apply only normalization.

    Returns:
        torch.Tensor: Transformed image tensor
    """

    mean = [0.5, 0.5, 0.5]
    std  = [0.5, 0.5, 0.5]

    if train:
        transform = T.Compose([
            T.RandomHorizontalFlip(p=0.5),
            T.RandomRotation(10),
            T.RandomCrop(28, padding=2),
            T.Normalize(mean=mean, std=std)
        ])
    else:
        transform = T.Compose([
            T.Normalize(mean=mean, std=std)
        ])

    X_out = []
    for i in range(X.shape[0]):
        X_out.append(transform(X[i]))

    return torch.stack(X_out)