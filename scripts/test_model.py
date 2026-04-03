import sys
from pathlib import Path

root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root))

from src.model import Mythos
from src.utils import load_config, get_device

def main():
    config = load_config("configs/model/debug.yaml")
    model_config = config["model"]
    
    model = Mythos(model_config)
    
    print("=" * 60)
    print("Mythos — Model Test")
    print("=" * 60)
    print(f"Model created successfully!")
    print(f"Parameters     : {model.get_num_params():,}")
    print(f"Device         : {get_device()}")
    print(f"Embedding      : {model.embedding}")
    print(f"Number of layers: {len(model.layers)}")
    print("=" * 60)

if __name__ == "__main__":
    main()