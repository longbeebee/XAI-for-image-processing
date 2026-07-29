"""Deterministic quality shifts in RGB pixel space."""

from __future__ import annotations

from io import BytesIO

from PIL import Image, ImageEnhance, ImageFilter


def apply_perturbation(image: Image.Image, kind: str, value: float | int) -> Image.Image:
    """Return a new RGB image with the requested quality perturbation."""
    source = image.convert("RGB").copy()
    if kind == "brightness":
        return ImageEnhance.Brightness(source).enhance(float(value))
    if kind == "blur":
        return source.filter(ImageFilter.GaussianBlur(radius=float(value)))
    if kind == "jpeg":
        buffer = BytesIO()
        source.save(buffer, format="JPEG", quality=int(value), optimize=False)
        buffer.seek(0)
        with Image.open(buffer) as decoded:
            return decoded.convert("RGB").copy()
    raise ValueError(f"Unsupported perturbation: {kind}")

