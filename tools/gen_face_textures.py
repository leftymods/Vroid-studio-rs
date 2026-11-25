#!/usr/bin/env python3
"""Generate the default VRoid-style face decal textures.

Each face decal mesh in the VRoid base avatar has its own 0..1 UV space, so a
single square texture maps 1:1 onto the patch.  We draw stylised anime features
(iris, sclera, lash, liner, brow, mouth) with alpha so the flat coloured patches
become proper features.
"""

import math
import os
import sys

from PIL import Image, ImageDraw, ImageFilter

S = 512
OUT = "/root/vroid/face_tex"


def new():
    return Image.new("RGBA", (S, S), (0, 0, 0, 0))


def lerp(a, b, t):
    return tuple(int(a[i] + (b[i] - a[i]) * t) for i in range(len(a)))


def radial_ellipse(draw, cx, cy, rx, ry, stops, steps=180):
    """Paint a radial gradient inside an ellipse. stops: [(t, rgba)]."""
    for i in range(steps, 0, -1):
        t = i / steps
        # find colour at t
        col = stops[-1][1]
        for j in range(len(stops) - 1):
            t0, c0 = stops[j]
            t1, c1 = stops[j + 1]
            if t0 <= t <= t1:
                u = 0.0 if t1 == t0 else (t - t0) / (t1 - t0)
                col = (*lerp(c0[:3], c1[:3], u), int(c0[3] + (c1[3] - c0[3]) * u))
                break
        else:
            col = stops[0][1] if t < stops[0][0] else stops[-1][1]
        draw.ellipse([cx - rx * t, cy - ry * t, cx + rx * t, cy + ry * t], fill=col)


def soft_blob(img, cx, cy, r, color):
    layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=color)
    layer = layer.filter(ImageFilter.GaussianBlur(r * 0.35))
    img.alpha_composite(layer)


def gen_iris():
    img = new()
    d = ImageDraw.Draw(img)
    c = S / 2
    r = S * 0.49
    stops = [
        (0.00, (10, 10, 10, 255)),
        (0.18, (24, 24, 24, 255)),
        (0.32, (78, 78, 78, 255)),
        (0.62, (150, 150, 150, 255)),
        (0.80, (118, 118, 118, 255)),
        (0.94, (52, 52, 52, 255)),
        (1.00, (30, 30, 30, 255)),
    ]
    radial_ellipse(d, c, c, r, r, stops)
    # radial streaks
    import random
    random.seed(7)
    for _ in range(70):
        a = random.uniform(0, math.tau)
        r0 = r * random.uniform(0.28, 0.4)
        r1 = r * random.uniform(0.72, 0.95)
        col = (160, 160, 160, random.randint(20, 60))
        d.line([c + math.cos(a) * r0, c + math.sin(a) * r0,
                c + math.cos(a) * r1, c + math.sin(a) * r1], fill=col, width=2)
    # pupil
    radial_ellipse(d, c, c, r * 0.24, r * 0.24,
                   [(0.0, (6, 6, 6, 255)), (0.8, (16, 16, 16, 255)), (1.0, (34, 34, 34, 255))], 60)
    # soft light at bottom
    soft_blob(img, c, c + r * 0.42, r * 0.5, (190, 190, 190, 45))
    return img


def gen_eyewhite():
    img = new()
    d = ImageDraw.Draw(img)
    # vertical gradient: shadowed under the lid (top) -> bright
    for y in range(S):
        t = y / S
        col = lerp((196, 198, 210), (255, 253, 250), min(1.0, t * 1.6))
        d.line([0, y, S, y], fill=(*col, 255))
    # soft pink corners
    soft_blob(img, S * 0.06, S * 0.5, S * 0.16, (240, 190, 190, 90))
    soft_blob(img, S * 0.94, S * 0.5, S * 0.16, (240, 190, 190, 90))
    return img


def gen_highlight():
    img = new()
    soft_blob(img, S * 0.36, S * 0.30, S * 0.17, (255, 255, 255, 235))
    soft_blob(img, S * 0.66, S * 0.72, S * 0.075, (255, 255, 255, 200))
    return img


def gen_eyeline():
    img = new()
    d = ImageDraw.Draw(img)
    # dark band hugging the lower/outer rim of the patch
    for i in range(26):
        a = i / 25
        col = (28, 16, 14, int(255 * (1 - a) ** 0.6))
        inset = 6 + i * 3
        d.arc([inset, inset, S - inset, S - inset], 200, 340, fill=col, width=3)
    return img.filter(ImageFilter.GaussianBlur(1.2))


def gen_eyelash():
    img = new()
    d = ImageDraw.Draw(img)
    # thick lash along the top edge, tapering to the outer corner
    pts = []
    for i in range(41):
        t = i / 40
        x = S * (0.08 + 0.84 * t)
        y = S * (0.30 + 0.10 * math.sin(math.pi * t) - 0.06 * t)
        pts.append((x, y))
    for w in range(18, 0, -1):
        alpha = int(255 * (w / 18) ** 0.5)
        d.line(pts, fill=(20, 12, 10, alpha), width=w, joint="curve")
    # a few individual lashes
    for k in range(7):
        t = 0.15 + 0.7 * k / 6
        x = S * (0.08 + 0.84 * t)
        y = S * (0.28 + 0.10 * math.sin(math.pi * t))
        d.line([x, y, x - 10, y - S * 0.10], fill=(20, 12, 10, 220), width=4)
    return img.filter(ImageFilter.GaussianBlur(0.8))


def gen_brow():
    img = new()
    d = ImageDraw.Draw(img)
    # tapered arch across the middle band
    top = []
    bot = []
    for i in range(41):
        t = i / 40
        x = S * (0.06 + 0.88 * t)
        arch = math.sin(math.pi * t)
        y = S * (0.52 - 0.10 * arch)
        top.append((x, y - S * (0.045 + 0.03 * arch)))
        bot.append((x, y + S * (0.030 + 0.02 * arch)))
    poly = top + bot[::-1]
    d.polygon(poly, fill=(58, 36, 26, 255))
    # soften the inner end
    soft_blob(img, S * 0.10, S * 0.50, S * 0.05, (58, 36, 26, 120))
    return img.filter(ImageFilter.GaussianBlur(1.6))


def gen_mouth():
    img = new()
    d = ImageDraw.Draw(img)
    # subtle upper lip line with a cupid's bow
    pts = []
    for i in range(41):
        t = i / 40
        x = S * (0.12 + 0.76 * t)
        y = S * (0.50 + 0.055 * math.sin(math.pi * t) - 0.03 * math.sin(math.pi * 2 * t))
        pts.append((x, y))
    for w in range(10, 0, -1):
        alpha = int(230 * (w / 10) ** 0.5)
        d.line(pts, fill=(150, 78, 74, alpha), width=w, joint="curve")
    # lower lip highlight
    soft_blob(img, S * 0.5, S * 0.60, S * 0.20, (232, 150, 145, 90))
    return img.filter(ImageFilter.GaussianBlur(1.0))


def gen_face():
    img = new()
    d = ImageDraw.Draw(img)
    # neutral base so the body skin colour tints face and body identically
    d.rectangle([0, 0, S, S], fill=(255, 255, 255, 255))
    # gentle cheek blush (rough placement on the face atlas)
    soft_blob(img, S * 0.24, S * 0.62, S * 0.11, (255, 150, 140, 70))
    soft_blob(img, S * 0.76, S * 0.62, S * 0.11, (255, 150, 140, 70))
    return img


def gen_debug():
    img = Image.new("RGBA", (S, S), (255, 255, 255, 255))
    d = ImageDraw.Draw(img)
    for i in range(0, S, 64):
        d.line([i, 0, i, S], fill=(180, 180, 180, 255))
        d.line([0, i, S, i], fill=(180, 180, 180, 255))
    d.rectangle([0, 0, S - 1, 24], fill=(255, 0, 0, 255))       # TOP
    d.rectangle([S - 24, 0, S - 1, S], fill=(0, 180, 0, 255))    # RIGHT
    d.rectangle([0, S - 24, S - 1, S - 1], fill=(0, 0, 255, 255))  # BOTTOM
    d.rectangle([0, 0, 24, S - 1], fill=(255, 200, 0, 255))      # LEFT
    d.ellipse([S * 0.35, S * 0.35, S * 0.65, S * 0.65], fill=(0, 0, 0, 255))
    d.text((S * 0.42, S * 0.46), "TOP", fill=(255, 255, 255, 255))
    return img


def main():
    os.makedirs(OUT, exist_ok=True)
    gens = {
        "iris": gen_iris,
        "eyewhite": gen_eyewhite,
        "highlight": gen_highlight,
        "eyeline": gen_eyeline,
        "eyelash": gen_eyelash,
        "brow": gen_brow,
        "mouth": gen_mouth,
        "face": gen_face,
        "debug": gen_debug,
    }
    only = sys.argv[1:] or list(gens)
    for name in only:
        img = gens[name]()
        path = os.path.join(OUT, f"{name}.png")
        img.save(path)
        print("wrote", path)


if __name__ == "__main__":
    main()
