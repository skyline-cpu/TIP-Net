"""
data/transforms.py  —  Image transforms for frame-level inputs.
"""

from torchvision import transforms


def get_train_transform(size: int = 224) -> transforms.Compose:
    """Augmented transform for training frames."""
    return transforms.Compose([
        transforms.Resize((size + 32, size + 32)),
        transforms.RandomResizedCrop(size, scale=(0.7, 1.0), ratio=(0.9, 1.1)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomApply([
            transforms.ColorJitter(brightness=0.2, contrast=0.2,
                                   saturation=0.2, hue=0.05),
        ], p=0.6),
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=5)], p=0.2),
        transforms.RandomGrayscale(p=0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


def get_val_transform(size: int = 224) -> transforms.Compose:
    """Deterministic transform for validation / test frames."""
    return transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


def get_xception_transform(size: int = 299, train: bool = True) -> transforms.Compose:
    """Xception uses [-1, 1] normalisation and 299×299 input."""
    if train:
        return transforms.Compose([
            transforms.Resize((size + 32, size + 32)),
            transforms.RandomResizedCrop(size, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([
                transforms.ColorJitter(0.2, 0.2, 0.2, 0.05)
            ], p=0.5),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])