from pathlib import Path
import yaml

def load_config(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def hotels_list(cfg) -> list[str]:
    return [h["name"] for h in cfg["hotels"]]

def websites(cfg) -> list[str]:
    return [w.upper() for w in cfg.get("websites", [])]

