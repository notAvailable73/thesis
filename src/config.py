from dataclasses import dataclass, field
from typing import List


@dataclass
class Config:
    # Reproducibility
    seed: int = 42

    # Dataset
    data_root: str = "data"
    image_size: int = 224
    num_classes: int = 5
    shots: int = 5
    query_per_class: int = 15
    test_class_ids: List[int] = field(default_factory=lambda: [
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
        10, 11, 12, 13, 14, 15, 16, 17, 18, 19
    ])

    # Model
    backbone: str = "resnet18"
    feature_dim: int = 512
    adapter_rank: int = 16

    # Training
    batch_size: int = 25
    lr: float = 5e-3
    num_steps: int = 300
    weight_decay: float = 0.0
    kl_anneal_steps: int = 300
    kl_weight_max: float = 0.1  # cap KL to avoid overwhelming MSE on small support set

    # OOD
    ood_dataset: str = "svhn"
    ood_num_samples: int = 500

    # Outputs
    checkpoint_dir: str = "checkpoints"
    results_dir: str = "results"


CFG = Config()
