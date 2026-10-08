from PIL import ImageFont


def get_font(size=16):
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", size)
    except OSError:
        font = ImageFont.load_default()

    return font
