import json
import logging
from pathlib import Path


logger = logging.getLogger("register_model")
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
logger.addHandler(handler)


ROOT_DIR = Path(__file__).resolve().parents[2]


def main() -> None:
    experiment_info_path = ROOT_DIR / "reports" / "experiment_info.json"
    with experiment_info_path.open("r") as file:
        experiment_info = json.load(file)

    accuracy = experiment_info.get("accuracy")
    logger.debug("Loaded experiment info from %s", experiment_info_path)
    logger.info("Model registration placeholder completed. Accuracy: %s", accuracy)


if __name__ == "__main__":
    main()
