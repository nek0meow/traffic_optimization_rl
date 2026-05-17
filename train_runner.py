import json
import sys

from src.main import run_training

if __name__ == "__main__":
    config = json.loads(sys.argv[1])
    run_training(config)