import os
from PIL import Image
import pandas as pd
from matplotlib import pyplot as plt, patches


def read_box_file(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split(",")
            coords = list(map(int, parts[:8]))
            text = ",".join(parts[8:])
            rows.append({
                "x1": coords[0],
                "y1": coords[1],
                "x2": coords[2],
                "y2": coords[3],
                "x3": coords[4],
                "y3": coords[5],
                "x4": coords[6],
                "y4": coords[7],
                "text": text
            })
    return pd.DataFrame(rows)

def draw_boxes(ocr_df: pd.DataFrame, image: Image.Image):
    fig, ax = plt.subplots(figsize=(14, 14))
    ax.imshow(image)
    for _, row in ocr_df.iterrows():
        x = row["x1"]
        y = row["y1"]
        width = row["x2"] - row["x1"]
        height = row["y4"] - row["y1"]
        rect = patches.Rectangle(
            (x, y),
            width,
            height,
            linewidth=1,
            edgecolor="red",
            facecolor="none"
        )
        ax.add_patch(rect)
        ax.text(
            x,
            y - 2,
            row["text"][:15],
            fontsize=6,
            color="blue"
        )
    plt.axis("off")
    plt.show()