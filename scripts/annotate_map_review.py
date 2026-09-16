"""把 terrain_anchors_vision_draft.json 的高置信标注点画到地图候选图上，供人工审查。

用法:
    PYTHONPATH=src python3 scripts/annotate_map_review.py

输出:
    data/S45/review/annotated_<image_id>.png
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DRAFT_JSON = (
    PROJECT_ROOT
    / "data/S45/normalized/terrain_anchors_vision_draft.json"
)
OUT_DIR = PROJECT_ROOT / "data/S45/review"
TARGET_WIDTH = 1400

FONT_CANDIDATES = [
    "/System/Library/Fonts/PingFang.ttc",
    "/System/Library/Fonts/STHeiti Medium.ttc",
    "/System/Library/Fonts/Supplemental/Songti.ttc",
    "/Library/Fonts/Arial Unicode.ttf",
]

COLOR_BY_KIND = {
    "fountain": (255, 200, 0),
    "tower": (255, 60, 60),
    "buff": (80, 220, 80),
    "dragon": (200, 80, 255),
    "extra": (255, 255, 255),
}


def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for candidate in FONT_CANDIDATES:
        if Path(candidate).exists():
            try:
                return ImageFont.truetype(candidate, size=size)
            except OSError:
                continue
    return ImageFont.load_default()


def _kind_of(anchor_id: str, *, is_extra: bool) -> str:
    if is_extra:
        return "extra"
    if "fountain" in anchor_id:
        return "fountain"
    if anchor_id.startswith("tower"):
        return "tower"
    if "buff" in anchor_id:
        return "buff"
    if anchor_id in {"tyrant", "overlord", "storm_dragon"}:
        return "dragon"
    return "extra"


def _draw_detection(
    draw: ImageDraw.ImageDraw,
    *,
    pos: tuple[int, int],
    anchor_id: str,
    confidence: float,
    color: tuple[int, int, int],
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
) -> None:
    x, y = pos
    r = 9
    draw.ellipse((x - r, y - r, x + r, y + r), outline=color, width=3)
    draw.line((x - r - 5, y, x + r + 5, y), fill=color, width=2)
    draw.line((x, y - r - 5, x, y + r + 5), fill=color, width=2)
    label = f"{anchor_id} {confidence:.2f}"
    lx = x + 12
    ly = y - 22
    draw.rectangle((lx - 2, ly - 2, lx + 8 * len(label) + 2, ly + 18), fill=(0, 0, 0))
    draw.text((lx, ly), label, fill=color, font=font)


def main() -> int:
    draft = json.loads(DRAFT_JSON.read_text(encoding="utf-8"))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    font = _font(16)
    legend_font = _font(20)

    outputs: list[Path] = []
    for image_entry in draft["per_image"]:
        image_path = PROJECT_ROOT / image_entry["file"]
        if not image_path.exists():
            print(f"跳过（文件缺失）: {image_path}", file=sys.stderr)
            continue

        detections = [
            det
            for det in image_entry["detections"] + image_entry.get("extra_detections", [])
            if det.get("position") is not None and det.get("confidence", 0.0) >= 0.6
        ]
        if not detections:
            print(f"{image_entry['image_id']}: 无高置信坐标，跳过")
            continue

        with Image.open(image_path) as source:
            base = source.convert("RGB")
        width, height = base.size
        scale = max(1.0, TARGET_WIDTH / width)
        canvas_size = (int(width * scale), int(height * scale))
        canvas = base.resize(canvas_size, Image.Resampling.LANCZOS)
        draw = ImageDraw.Draw(canvas)

        for det in detections:
            raw_pos = det["position"]
            pos = (int(raw_pos[0] * scale), int(raw_pos[1] * scale))
            is_extra = det in image_entry.get("extra_detections", [])
            color = COLOR_BY_KIND[_kind_of(det["anchor_id"], is_extra=is_extra)]
            _draw_detection(
                draw,
                pos=pos,
                anchor_id=det["anchor_id"],
                confidence=float(det["confidence"]),
                color=color,
                font=font,
            )

        title = (
            f"{image_entry['image_id']}  {width}x{height}  "
            f"高置信标注 {len(detections)} 项 | draft pending review"
        )
        draw.rectangle((0, 0, canvas_size[0], 40), fill=(0, 0, 0))
        draw.text((10, 8), title, fill=(255, 255, 255), font=legend_font)
        legend_y = 52
        for kind, (r, g, b) in COLOR_BY_KIND.items():
            draw.ellipse((12, legend_y + 4, 24, legend_y + 16), outline=(r, g, b), width=3)
            draw.text((32, legend_y), kind, fill=(r, g, b), font=font)
            legend_y += 26

        target = OUT_DIR / f"annotated_{image_entry['image_id']}.png"
        canvas.save(target)
        outputs.append(target)
        print(f"已生成: {target}")

    print(f"共生成 {len(outputs)} 张审查图")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
