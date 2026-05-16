from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SUMMARY_PATH = PROJECT_ROOT / "outputs" / "matching" / "sift_matching_summary.csv"
OUTPUT_PATH = PROJECT_ROOT / "outputs" / "figures" / "sift_good_matches_chart.png"


def load_matching_summary(csv_path: Path) -> pd.DataFrame:
    """Load the SIFT matching summary CSV."""
    if not csv_path.exists():
        raise FileNotFoundError(f"Matching summary not found: {csv_path}")

    return pd.read_csv(csv_path)


def save_good_matches_chart(summary_df: pd.DataFrame, output_path: Path) -> None:
    """Create and save a bar chart of good SIFT matches by augmentation."""
    fig, ax = plt.subplots(figsize=(8, 5))

    bars = ax.bar(
        summary_df["augmentation"],
        summary_df["good_matches"],
        color=["#4C78A8", "#F58518", "#54A24B"],
    )

    ax.set_title("SIFT Good Matches by Augmentation")
    ax.set_xlabel("Augmentation")
    ax.set_ylabel("Good Matches")

    # Add exact match counts above each bar for easier comparison.
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            f"{int(height)}",
            ha="center",
            va="bottom",
        )

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def print_match_extremes(summary_df: pd.DataFrame) -> None:
    """Print augmentations with the highest and lowest good match counts."""
    highest_row = summary_df.loc[summary_df["good_matches"].idxmax()]
    lowest_row = summary_df.loc[summary_df["good_matches"].idxmin()]

    print(
        "Augmentation with highest matches: "
        f"{highest_row['augmentation']} ({highest_row['good_matches']})"
    )
    print(
        "Augmentation with lowest matches: "
        f"{lowest_row['augmentation']} ({lowest_row['good_matches']})"
    )


def main() -> None:
    summary_df = load_matching_summary(SUMMARY_PATH)
    save_good_matches_chart(summary_df, OUTPUT_PATH)
    print_match_extremes(summary_df)
    print(f"Saved chart to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
