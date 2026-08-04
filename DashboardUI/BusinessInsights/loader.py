import json
from pathlib import Path

# Directory containing this loader.py
BASE_DIR = Path(__file__).resolve().parent


def load_insights(filename: str) -> dict:
    """
    Load a business insights JSON file.

    Parameters
    ----------
    filename : str
        Example:
            "kmeans_insights.json"
            "dbscan_insights.json"

    Returns
    -------
    dict
        Parsed JSON content.
    """

    filepath = BASE_DIR / filename

    if not filepath.exists():
        raise FileNotFoundError(
            f"Insight file not found:\n{filepath}"
        )

    with filepath.open("r", encoding="utf-8") as f:
        return json.load(f)


# Load once when the module is imported
KMEANS_INSIGHTS = load_insights("kmeans_insights.json")
DBSCAN_INSIGHTS = load_insights("dbscan_insights.json")