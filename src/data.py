import random
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms


def get_cifar100_test(data_root: str, image_size: int) -> Dataset:
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    return datasets.CIFAR100(root=data_root, train=False,
                             download=True, transform=transform)


def sample_episode(dataset, class_ids, num_classes, shots, query_per_class, seed):
    rng = random.Random(seed)
    chosen = rng.sample(class_ids, num_classes)

    # Build index: label -> list of dataset indices
    label_to_indices = {}
    for idx, (_, label) in enumerate(dataset):
        if label not in label_to_indices:
            label_to_indices[label] = []
        label_to_indices[label].append(idx)

    support_images, support_labels = [], []
    query_images, query_labels = [], []

    for new_label, orig_label in enumerate(chosen):
        indices = label_to_indices[orig_label]
        rng.shuffle(indices)
        needed = shots + query_per_class
        assert len(indices) >= needed, (
            f"Class {orig_label} has only {len(indices)} samples, need {needed}"
        )
        for i in indices[:shots]:
            support_images.append(dataset[i][0])
            support_labels.append(new_label)
        for i in indices[shots:shots + query_per_class]:
            query_images.append(dataset[i][0])
            query_labels.append(new_label)

    support_x = torch.stack(support_images)
    support_y = torch.tensor(support_labels, dtype=torch.long)
    query_x = torch.stack(query_images)
    query_y = torch.tensor(query_labels, dtype=torch.long)
    return support_x, support_y, query_x, query_y


def get_svhn_ood(data_root: str, image_size: int, num_samples: int, seed: int) -> torch.Tensor:
    import os
    svhn_root = os.path.join(data_root, "svhn")
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    dataset = datasets.SVHN(root=svhn_root, split="test",
                            download=True, transform=transform)
    rng = random.Random(seed)
    indices = rng.sample(range(len(dataset)), min(num_samples, len(dataset)))
    images = [dataset[i][0] for i in indices]
    return torch.stack(images)


if __name__ == "__main__":
    from src.config import CFG

    print("Loading CIFAR-100 test set...")
    dataset = get_cifar100_test(CFG.data_root, CFG.image_size)
    print(f"  Total samples: {len(dataset)}")

    print("Sampling episode...")
    sx, sy, qx, qy = sample_episode(
        dataset, CFG.test_class_ids, CFG.num_classes,
        CFG.shots, CFG.query_per_class, CFG.seed
    )
    print(f"  support_x: {sx.shape}  (expect (25, 3, 224, 224))")
    print(f"  support_y: {sy.shape}  values: {sy.tolist()}")
    print(f"  query_x:   {qx.shape}  (expect (75, 3, 224, 224))")
    assert sx.shape == (25, 3, 224, 224)
    assert qx.shape == (75, 3, 224, 224)

    print("Loading SVHN OOD...")
    svhn = get_svhn_ood(CFG.data_root, CFG.image_size, CFG.ood_num_samples, CFG.seed)
    print(f"  svhn_x: {svhn.shape}  (expect (500, 3, 224, 224))")
    assert svhn.shape == (500, 3, 224, 224)

    print("ALL TESTS PASSED")
