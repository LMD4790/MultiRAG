from __future__ import annotations

import shutil
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import DATASETS_DIR, GRAPHRAG_ROOT, load_project_env


CSV_COLUMNS = {
    "destination": "目的地",
    "transport": "交通安排",
    "accommodation": "住宿推荐",
    "attractions": "必打卡景点",
    "food": "美食推荐",
    "tips": "实用小贴士",
    "thoughts": "旅行感悟",
}


def read_travel_guide_csv() -> pd.DataFrame:
    csv_path = DATASETS_DIR / "travel_guide.csv"
    try:
        return pd.read_csv(csv_path, encoding="gbk")
    except UnicodeDecodeError:
        return pd.read_csv(csv_path, encoding="utf-8")


def build_travel_guide_text(travel_dataframe: pd.DataFrame) -> str:
    missing = [column for column in CSV_COLUMNS.values() if column not in travel_dataframe.columns]
    if missing:
        raise ValueError(f"travel_guide.csv is missing columns: {missing}")

    paragraphs: list[str] = []
    for _, row in travel_dataframe.iterrows():
        paragraphs.append(
            f"{row[CSV_COLUMNS['destination']]}旅游攻略："
            f"交通安排：{row[CSV_COLUMNS['transport']]}\n"
            f"住宿推荐：{row[CSV_COLUMNS['accommodation']]}\n"
            f"必打卡景点：{row[CSV_COLUMNS['attractions']]}\n"
            f"美食推荐：{row[CSV_COLUMNS['food']]}\n"
            f"实用小贴士：{row[CSV_COLUMNS['tips']]}\n"
            f"旅行感悟：{row[CSV_COLUMNS['thoughts']]}\n"
        )
    return "\n".join(paragraphs)


def main() -> None:
    load_project_env()
    travel_dataframe = read_travel_guide_csv()
    travel_text = build_travel_guide_text(travel_dataframe)

    dataset_output = DATASETS_DIR / "travel_guide.txt"
    dataset_output.write_text(travel_text, encoding="utf-8")

    graphrag_input_dir = GRAPHRAG_ROOT / "input"
    graphrag_input_dir.mkdir(parents=True, exist_ok=True)
    graphrag_input = graphrag_input_dir / "travel_guide.txt"
    shutil.copyfile(dataset_output, graphrag_input)

    print(f"Wrote {dataset_output}")
    print(f"Copied GraphRAG input to {graphrag_input}")
    print("Next step: graphrag index --root GraphRAG/tourist_graphrag")


if __name__ == "__main__":
    main()
