import io
import random

import numpy as np
import streamlit as st

from PIL import Image, ImageDraw, ImageFont, ImageOps
from typing import List, Tuple

CAPTIONS: List[str] = [
    "When the code works but you don't know why",
    "Unit tests?  More like guesses.",
    "Commit code, cross fingers",
    "Did you turn it off and on again?",
    "Monday: 0 bugs found.  Tuesday: 27 new features",
    "I don't always write Python, but when I do, it works on the first try",
    "404: caffeine not found",
    "Debugging: being the detective in a crime movie where you are also the murderer",
    "When you realise it was a semicolon all along",
    "That moment you forget to push before the meeting"
]

def generate_random_pastel_color(brightness: float) -> Tuple[int, int, int]:
    """Generate a pastel colour scaled by brightness for mosaic blocks."""
    base = np.random.uniform(180, 255, 3)
    scaled = base * brightness
    return tuple(int(min(255, max(0, c))) for c in scaled)


def generate_mosaic_pattern(template: Image.Image, block_size: int) -> Image.Image:
    width, height = template.size
    grayscale = np.array(ImageOps.grayscale(template), dtype=np.float32) / 255.0
    mosaic = Image.new("RGB", template.size)
    draw = ImageDraw.Draw(mosaic)
    for y in range(0, height, block_size):
        for x in range(0, width, block_size):
            x_end = min(x + block_size, width)
            y_end = min(y + block_size, height)
            block = grayscale[y:y_end, x:x_end]
            brightness = float(block.mean()) if block.size > 0 else 0.5
            colour = generate_random_pastel_color(brightness)
            draw.rectangle([x, y, x_end, y_end], fill=colour)
    return mosaic


def apply_mosaic_overlay(template: Image.Image, block_size: int = 12, alpha: float = 0.5) -> Image.Image:
    mosaic = generate_mosaic_pattern(template, block_size)
    base_arr = np.array(template.convert("RGB"), dtype=np.float32)
    mosaic_arr = np.array(mosaic, dtype=np.float32)
    blended = (1.0 - alpha) * base_arr + alpha * mosaic_arr
    blended = np.clip(blended, 0, 255).astype(np.uint8)
    return Image.fromarray(blended, mode="RGB")


def apply_glitch_overlay(template: Image.Image, max_shift: int = 10, alpha: float = 0.5) -> Image.Image:
    arr = np.array(template.convert("RGB"), dtype=np.float32)
    h, w, c = arr.shape
    glitched = np.empty_like(arr)
    for channel in range(c):
        for row in range(h):
            shift = np.random.randint(-max_shift, max_shift + 1)
            glitched[row, :, channel] = np.roll(arr[row, :, channel], shift)
    blended = (1.0 - alpha) * arr + alpha * glitched
    blended = np.clip(blended, 0, 255).astype(np.uint8)
    return Image.fromarray(blended, mode="RGB")


def wrap_text(text: str, font: ImageFont.FreeTypeFont, max_width: int) -> List[str]:
    lines: List[str] = []
    words = text.split()
    while words:
        line_words: List[str] = []
        while words:
            line_words.append(words.pop(0))
            test_line = " ".join(line_words + words[:1])
            bbox = font.getbbox(test_line)
            line_width = bbox[2] - bbox[0]
            if line_width > max_width:
                break
        lines.append(" ".join(line_words))
    return lines


def add_caption(
    image: Image.Image,
    caption: str,
    font_path: str | None = None,
    margin: int = 20,
) -> Image.Image:
    img = image.copy()
    font_size = max(20, img.height // 16)
    try:
        font = ImageFont.truetype(font_path or "DejaVuSans-Bold.ttf", font_size)
    except Exception:
        font = ImageFont.load_default()
    draw = ImageDraw.Draw(img)
    max_text_width = img.width - 2 * margin
    lines = wrap_text(caption, font, max_text_width)
    bbox_hg = font.getbbox("Hg")
    line_height = (bbox_hg[3] - bbox_hg[1]) + 4
    text_block_height = line_height * len(lines)
    y_start = img.height - text_block_height - margin
    shadow_offset = (2, 2)
    for i, line in enumerate(lines):
        bbox_line = font.getbbox(line)
        w = bbox_line[2] - bbox_line[0]
        x = (img.width - w) // 2
        y = y_start + i * line_height
        draw.text((x + shadow_offset[0], y + shadow_offset[1]), line, font=font, fill="black")
        draw.text((x, y), line, font=font, fill="white")
    return img


def main() -> None:
    st.title("🎨 Creative Meme Generator")
    st.write(
        "Upload a template image, pick a caption and adjust the overlay intensity "
        "to create your own unique meme."
    )

    st.sidebar.header("Settings")
    overlay_style = st.sidebar.selectbox(
        "Overlay style",
        options=["Mosaic", "Glitch", "None"],
        index=0,
        help="Choose how to transform your template before adding the caption."
    )
    alpha = st.sidebar.slider(
        "Overlay intensity",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.05,
        help="Controls how strongly the selected overlay is blended with your template."
    )
    block_size = None
    max_shift = None
    if overlay_style == "Mosaic":
        block_size = st.sidebar.slider(
            "Mosaic block size",
            min_value=4,
            max_value=64,
            value=16,
            step=2,
            help="Size of each square in the mosaic overlay. Smaller values produce finer patterns."
        )
    elif overlay_style == "Glitch":
        max_shift = st.sidebar.slider(
            "Maximum shift (pixels)",
            min_value=1,
            max_value=80,
            value=20,
            step=1,
            help="Maximum number of pixels rows are shifted left or right to create the glitch effect."
        )
    use_random_caption = st.sidebar.checkbox(
        "Use random caption from list",
        value=True,
        help="If checked, the app will pick a caption from its built‑in list."
    )
    user_caption = ""
    if not use_random_caption:
        user_caption = st.sidebar.text_input(
            "Enter your caption",
            value="",
            help="Type your own caption here (leave blank to use a random one).",
        )

    uploaded_file = st.file_uploader(
        "Choose an image file",
        type=["jpg", "jpeg", "png"],
        help="Upload the meme template you want to work with."
    )

    if uploaded_file is not None:
        template = Image.open(uploaded_file).convert("RGB")
        if overlay_style == "Mosaic":
            if block_size is None:
                bs = max(4, template.width // 40)
            else:
                bs = block_size
            blended = apply_mosaic_overlay(template, block_size=bs, alpha=alpha)
        elif overlay_style == "Glitch":
            if max_shift is None:
                ms = max(1, template.width // 40)
            else:
                ms = max_shift
            blended = apply_glitch_overlay(template, max_shift=ms, alpha=alpha)
        else:
            blended = template.copy()
        caption_text = user_caption.strip() if (user_caption and not use_random_caption) else random.choice(CAPTIONS)
        result = add_caption(blended, caption_text)
        st.image(result, caption="Your generated meme", use_column_width=True)
        buffer = io.BytesIO()
        result.save(buffer, format="JPEG")
        btn = st.download_button(
            label="Download meme",
            data=buffer.getvalue(),
            file_name="creative_meme.jpg",
            mime="image/jpeg",
        )
    else:
        st.info("Upload a template image to get started.")


if __name__ == "__main__":
    main()