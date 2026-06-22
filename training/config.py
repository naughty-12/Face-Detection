"""Training hyperparameter management and auto-computation"""
import os
import yaml

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
CONFIG_PATH = os.path.join(PROJECT_ROOT, "model", "model_config.yaml")


def load_config():
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)


def compute_warmup_steps(epochs, dataset_size, batch_size):
    steps_per_epoch = dataset_size // batch_size
    warmup_steps = int(epochs * steps_per_epoch * 0.05)
    return max(warmup_steps, 500)


def print_training_summary(config, dataset_size):
    cfg = config["training"]
    batch_size = cfg["batch_size"]
    epochs = cfg["epochs"]
    steps_per_epoch = dataset_size // batch_size
    warmup_steps = compute_warmup_steps(epochs, dataset_size, batch_size)
    total_steps = epochs * steps_per_epoch

    print("=" * 60)
    print("Training Configuration Summary")
    print("=" * 60)
    print(f"  Model:          {config['model_name']}")
    print(f"  Input Size:     {config['input_size']}")
    print(f"  Batch Size:     {batch_size}")
    print(f"  Epochs:         {epochs}")
    print(f"  Steps/Epoch:    {steps_per_epoch}")
    print(f"  Total Steps:    {total_steps}")
    print(f"  Warmup Steps:   {warmup_steps}  (>=500 ensured)")
    print(f"  Optimizer:      {cfg['optimizer']}")
    print(f"  Initial LR:     {cfg['lr0']}")
    print(f"  Final LR:       {cfg['lrf']}")
    print(f"  Mixed Precision: {cfg['amp']}")
    print(f"  Dataset Size:   {dataset_size}")
    print("=" * 60)


WIDER_TRAIN_SIZE = 12880

if __name__ == "__main__":
    config = load_config()
    print_training_summary(config, WIDER_TRAIN_SIZE)
