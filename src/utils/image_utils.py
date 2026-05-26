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

def visualize_token_grid(image_path, token_count):
    """
    Draws the Qwen2-VL structural token grid over the receipt image.
    """
    img = Image.open(image_path)
    W, H = img.size
    
    # Qwen2-VL treats a 28x28 pixel block as 1 unified token representation
    token_pixel_size = 28
    
    # Calculate how many token rows and columns fit into this image geometry
    cols = W // token_pixel_size
    rows = H // token_pixel_size
    
    fig, ax = plt.subplots(figsize=(10, 12))
    ax.imshow(img)
    
    # Draw a bounding box for every single visual token block
    for r in range(rows):
        for c in range(cols):
            # Each box represents the boundary of ONE visual token vector
            rect = patches.Rectangle(
                (c * token_pixel_size, r * token_pixel_size), 
                token_pixel_size, 
                token_pixel_size, 
                linewidth=0.5, 
                edgecolor='red', 
                facecolor='none',
                alpha=0.4
            )
            ax.add_patch(rect)
            
    plt.title(f"Original Size: {W}x{H} | Visualized Tokens: ~{rows * cols} (Actual: {token_count})")
    plt.axis('off')
    plt.show()