import json, yaml
from pathlib import Path

def load_config(config_path):
    with open(Path(config_path)) as file:
        return yaml.safe_load(file)
    
def load_history(history_path):
    with open(Path(history_path)) as file:
        return json.load(file)

def save_history(history_path, history):
    with open(Path(history_path), "w") as file:
        json.dump(history, file)