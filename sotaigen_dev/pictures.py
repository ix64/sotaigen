"""What the review sheets and study pictures share: the font their labels are drawn in."""

from PIL import ImageFont

# Chinese labels need a face with CJK; a machine without these gets Pillow's own, which scales.
FACES = ("/System/Library/Fonts/STHeiti Medium.ttc", "/System/Library/Fonts/STHeiti Light.ttc", "DejaVuSans.ttf")


def font(size=14):
    for name in FACES:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default(size)
