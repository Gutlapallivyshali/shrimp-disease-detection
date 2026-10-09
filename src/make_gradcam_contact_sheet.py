from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import math

PROJECT_ROOT = Path.home() / "Documents" / "shrimp-disease-detection"

INPUT_DIR = PROJECT_ROOT / "outputs" / "gradcam" / "original_only_errors"
OUTPUT_FILE = PROJECT_ROOT / "outputs" / "gradcam" / "original_only_errors_contact_sheet.jpg"

files = sorted(INPUT_DIR.glob("*.jpg"))

if not files:
    print("No Grad-CAM++ images found.")
    raise SystemExit

# Size of each image area
CELL_WIDTH = 500
CELL_HEIGHT = 330

# 2 images per row
COLUMNS = 2
ROWS = math.ceil(len(files) / COLUMNS)

# Create blank contact sheet
sheet = Image.new(
    "RGB",
    (COLUMNS * CELL_WIDTH, ROWS * CELL_HEIGHT),
    "white"
)

draw = ImageDraw.Draw(sheet)

# Use default font
try:
    font = ImageFont.truetype("arial.ttf", 18)
except:
    font = ImageFont.load_default()

for index, file_path in enumerate(files):

    image = Image.open(file_path).convert("RGB")

    # Keep original aspect ratio
    image.thumbnail((CELL_WIDTH - 20, 260))

    column = index % COLUMNS
    row = index // COLUMNS

    x = column * CELL_WIDTH + (CELL_WIDTH - image.width) // 2
    y = row * CELL_HEIGHT + 5

    sheet.paste(image, (x, y))

    # File name below image
    label = file_path.stem

    text_x = column * CELL_WIDTH + 10
    text_y = row * CELL_HEIGHT + 275

    draw.text(
        (text_x, text_y),
        label,
        fill="black",
        font=font
    )

sheet.save(OUTPUT_FILE, quality=95)

print()
print("Contact sheet created successfully!")
print()
print("Saved to:")
print(OUTPUT_FILE)