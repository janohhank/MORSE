from dataclasses import dataclass


@dataclass
class TrainingConfig:
    # Seed for reproducibility
    seed: int
    # Bigger population -> bigger search space
    pop_size: int = 188
    # Bigger generation number -> bigger convergence
    ngen: int = 285
    # Crossing probability
    cxpb: float = 0.5
    # Mutation probability
    mutpb: float = 0.2
    # Use ROC-AUC instead of PR-AUC
    use_roc_auc: bool = True
    # Directory to save results
    result_directory: str = ""