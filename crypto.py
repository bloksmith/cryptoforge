import os
import random
import math
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageEnhance

def create_image(seed, output_folder):
    random.seed(seed)

    # Set image dimensions
    width, height = 400, 400

    # Create a new image
    img = Image.new("RGBA", (width, height), (0, 0, 0, 255))

    # Draw a line
    draw = ImageDraw.Draw(img)
    draw.line((0, height // 2, width, height // 2), fill=(255, 255, 255, 128), width=2)

    # Draw a spiral
    steps = 800
    step_size = 0.05
    for i in range(steps):
        t = i * step_size
        x = width // 2 + int((t * math.cos(t)) * width // 2.2)
        y = height // 2 + int((t * math.sin(t)) * height // 2.2)

        r = random.randint(0, 255)
        g = random.randint(0, 255)
        b = random.randint(0, 255)

        draw.ellipse((x - 1, y - 1, x + 1, y + 1), fill=(r, g, b, 255))

    # Save the image
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    img.save(os.path.join(output_folder, f"nft_seed{seed}.png"))

if __name__ == "__main__":
    create_image(42, "output")

