from __future__ import annotations

import cv2
import numpy as np

from vision.debug_utils import DebugOverlay
from vision.model_utils import DetectedObject


# ---------------- Yellow block detector, kept from original code ----------------

def detect_yellow_block(
    frame_bgr: np.ndarray,
) -> tuple[list[DetectedObject], list[DebugOverlay]]:
    """Detect a simple yellow block and return detections plus debug contours."""
    detections: list[DetectedObject] = []
    debug_overlays: list[DebugOverlay] = []

    blurred = cv2.GaussianBlur(frame_bgr, (5, 5), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    yellow_hsv_low = (20, 110, 80)
    yellow_hsv_high = (38, 255, 255)
    mask = cv2.inRange(hsv, yellow_hsv_low, yellow_hsv_high)

    open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel)

    min_area_px = 500
    min_fill_ratio = 0.30

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        contour_area = float(cv2.contourArea(contour))
        if contour_area < min_area_px:
            continue

        x, y, width, height = cv2.boundingRect(contour)
        bounding_box_area = float(max(1, width * height))
        fill_ratio = contour_area / bounding_box_area

        if fill_ratio < min_fill_ratio:
            continue

        confidence = yellow_detection_score(contour_area, min_area_px, fill_ratio)

        detection = DetectedObject(
            class_name="yellow block",
            confidence=confidence,
            x=int(x),
            y=int(y),
            width=int(width),
            height=int(height),
        )
        detection.add_attribute("color", "yellow", 1.0)

        detections.append(detection)
        debug_overlays.append(
            DebugOverlay(
                color=(0, 255, 255),
                contour=contour,
                label="yellow block",
                x=int(x),
                y=int(y),
            )
        )

    return detections, debug_overlays


def yellow_detection_score(
    contour_area: float,
    min_area_px: int,
    fill_ratio: float,
) -> float:
    area_score = min(1.0, contour_area / float(max(1, min_area_px * 4)))
    score = 0.55 * area_score + 0.45 * max(0.0, min(1.0, fill_ratio))
    return max(0.0, min(1.0, score))


# ---------------- Utility for clean rectangular debug boxes ----------------

def make_rect_contour(
    x: int,
    y: int,
    width: int,
    height: int,
) -> np.ndarray:
    return np.array(
        [
            [[x, y]],
            [[x + width, y]],
            [[x + width, y + height]],
            [[x, y + height]],
        ],
        dtype=np.int32,
    )


# ---------------- Trash can detector ----------------

def detect_trash_cans(
    frame_bgr: np.ndarray,
) -> tuple[list[DetectedObject], list[DebugOverlay]]:
    """
    Detect blue and green trash cans.

    New setup:
    - Blue trash can: blue body + black tape band
    - Green trash can: red tape marker + green body + black tape band

    The red marker is used to prevent the green can from being confused as blue.
    """
    detections: list[DetectedObject] = []
    debug_overlays: list[DebugOverlay] = []

    blurred = cv2.GaussianBlur(frame_bgr, (7, 7), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    green_mask = build_neon_green_mask(blurred, hsv)
    blue_mask = build_blue_mask(hsv)
    red_mask = build_red_marker_mask(hsv)

    green_detections, green_overlays = detect_green_trash_can_with_red_marker(
        frame_hsv=hsv,
        green_mask=green_mask,
        red_mask=red_mask,
        overlay_color=(0, 255, 0),
        min_area_px=700,
        min_confidence=0.32,
    )

    blue_detections, blue_overlays = detect_blue_trash_can(
        frame_hsv=hsv,
        blue_mask=blue_mask,
        red_mask=red_mask,
        overlay_color=(255, 0, 0),
        min_area_px=900,
        min_confidence=0.36,
    )

    detections.extend(green_detections)
    detections.extend(blue_detections)

    debug_overlays.extend(green_overlays)
    debug_overlays.extend(blue_overlays)

    return detections, debug_overlays


def build_neon_green_mask(
    frame_bgr: np.ndarray,
    frame_hsv: np.ndarray,
) -> np.ndarray:
    """
    Better mask for bright/neon green plastic.
    Combines HSV green and green-channel dominance.
    """
    hsv_green = cv2.inRange(
        frame_hsv,
        np.array((30, 20, 35), dtype=np.uint8),
        np.array((105, 255, 255), dtype=np.uint8),
    )

    b, g, r = cv2.split(frame_bgr)

    green_dominance = (
        (g > 70)
        & (g > r * 1.03)
        & (g > b * 1.03)
    ).astype(np.uint8) * 255

    mask = cv2.bitwise_or(hsv_green, green_dominance)

    open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))

    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel)

    return mask


def build_blue_mask(
    frame_hsv: np.ndarray,
) -> np.ndarray:
    """Stricter blue mask to reduce false blue detections."""
    mask = cv2.inRange(
        frame_hsv,
        np.array((90, 55, 35), dtype=np.uint8),
        np.array((135, 255, 255), dtype=np.uint8),
    )

    open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))

    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel)

    return mask


def build_red_marker_mask(
    frame_hsv: np.ndarray,
) -> np.ndarray:
    """
    Red tape marker mask.

    Red wraps around the HSV hue range, so it uses two masks.
    """
    red_low_1 = cv2.inRange(
        frame_hsv,
        np.array((0, 70, 45), dtype=np.uint8),
        np.array((12, 255, 255), dtype=np.uint8),
    )

    red_low_2 = cv2.inRange(
        frame_hsv,
        np.array((165, 70, 45), dtype=np.uint8),
        np.array((180, 255, 255), dtype=np.uint8),
    )

    mask = cv2.bitwise_or(red_low_1, red_low_2)

    open_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3))
    close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 7))

    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel)

    return mask


def build_black_band_mask(
    frame_hsv: np.ndarray,
) -> np.ndarray:
    """Black tape band mask."""
    mask = cv2.inRange(
        frame_hsv,
        np.array((0, 0, 0), dtype=np.uint8),
        np.array((180, 255, 135), dtype=np.uint8),
    )

    open_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 3))
    close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (35, 9))

    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel)

    return mask


def detect_green_trash_can_with_red_marker(
    frame_hsv: np.ndarray,
    green_mask: np.ndarray,
    red_mask: np.ndarray,
    overlay_color: tuple[int, int, int],
    min_area_px: int = 700,
    min_confidence: float = 0.32,
) -> tuple[list[DetectedObject], list[DebugOverlay]]:
    """
    Green trash can detector.

    The green can must have:
    - green body
    - red marker
    - black band score helps but is not mandatory
    """
    detections: list[DetectedObject] = []
    debug_overlays: list[DebugOverlay] = []

    frame_height, frame_width = frame_hsv.shape[:2]
    frame_area = float(frame_width * frame_height)
    black_mask = build_black_band_mask(frame_hsv)

    # Connect green regions across black/red tape.
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 95))
    connected_green = cv2.morphologyEx(green_mask, cv2.MORPH_CLOSE, vertical_kernel)

    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17))
    connected_green = cv2.morphologyEx(connected_green, cv2.MORPH_CLOSE, close_kernel)

    contours, _ = cv2.findContours(connected_green, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    valid_candidates: list[
        tuple[float, np.ndarray, int, int, int, int, float, float]
    ] = []

    for contour in contours:
        contour_area = float(cv2.contourArea(contour))
        if contour_area < min_area_px:
            continue

        x, y, width, height = cv2.boundingRect(contour)

        if width < 25 or height < 35:
            continue

        object_fraction = contour_area / frame_area
        if object_fraction > 0.90:
            continue

        aspect_ratio = height / float(max(1, width))
        if aspect_ratio < 0.35 or aspect_ratio > 3.40:
            continue

        center_y = y + height / 2.0
        vertical_position = center_y / float(frame_height)
        if vertical_position < 0.08:
            continue

        green_score = mask_fraction(green_mask, x, y, width, height)
        red_score = marker_score(red_mask, x, y, width, height)
        black_score = marker_score(black_mask, x, y, width, height)

        # The red marker is the key differentiator for the green can.
        if red_score < 0.015:
            continue

        confidence = green_trash_can_score(
            green_score=green_score,
            red_score=red_score,
            black_score=black_score,
            aspect_ratio=aspect_ratio,
            vertical_position=vertical_position,
            contour_area=contour_area,
            min_area_px=min_area_px,
        )

        if confidence < min_confidence:
            continue

        valid_candidates.append(
            (
                confidence,
                contour,
                int(x),
                int(y),
                int(width),
                int(height),
                red_score,
                black_score,
            )
        )

    if valid_candidates:
        valid_candidates = [max(valid_candidates, key=lambda item: item[0])]

    for confidence, _, x, y, width, height, red_score, black_score in valid_candidates:
        detection = DetectedObject(
            class_name="trash can",
            confidence=confidence,
            x=x,
            y=y,
            width=width,
            height=height,
        )
        detection.add_attribute("color", "green", 1.0)
        detection.add_attribute("marker", "red tape", red_score)
        detection.add_attribute("black_band", "detected" if black_score > 0.02 else "weak", black_score)

        detections.append(detection)

        rect_contour = make_rect_contour(x, y, width, height)
        debug_overlays.append(
            DebugOverlay(
                color=overlay_color,
                contour=rect_contour,
                label="green trash can",
                x=x,
                y=y,
            )
        )

    return detections, debug_overlays


def detect_blue_trash_can(
    frame_hsv: np.ndarray,
    blue_mask: np.ndarray,
    red_mask: np.ndarray,
    overlay_color: tuple[int, int, int],
    min_area_px: int = 900,
    min_confidence: float = 0.36,
) -> tuple[list[DetectedObject], list[DebugOverlay]]:
    """
    Blue trash can detector.

    The blue can should be blue and should NOT have a strong red marker.
    This prevents the green can with red tape from being misread as blue.
    """
    detections: list[DetectedObject] = []
    debug_overlays: list[DebugOverlay] = []

    frame_height, frame_width = frame_hsv.shape[:2]
    frame_area = float(frame_width * frame_height)
    black_mask = build_black_band_mask(frame_hsv)

    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 85))
    connected_blue = cv2.morphologyEx(blue_mask, cv2.MORPH_CLOSE, vertical_kernel)

    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17))
    connected_blue = cv2.morphologyEx(connected_blue, cv2.MORPH_CLOSE, close_kernel)

    contours, _ = cv2.findContours(connected_blue, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    valid_candidates: list[
        tuple[float, np.ndarray, int, int, int, int, float, float]
    ] = []

    for contour in contours:
        contour_area = float(cv2.contourArea(contour))
        if contour_area < min_area_px:
            continue

        x, y, width, height = cv2.boundingRect(contour)

        if width < 25 or height < 35:
            continue

        object_fraction = contour_area / frame_area
        if object_fraction > 0.90:
            continue

        aspect_ratio = height / float(max(1, width))
        if aspect_ratio < 0.35 or aspect_ratio > 3.40:
            continue

        center_y = y + height / 2.0
        vertical_position = center_y / float(frame_height)
        if vertical_position < 0.08:
            continue

        blue_score = mask_fraction(blue_mask, x, y, width, height)
        red_score = marker_score(red_mask, x, y, width, height)
        black_score = marker_score(black_mask, x, y, width, height)

        # If it has strong red marker, it is probably the green can, not blue.
        if red_score > 0.04:
            continue

        confidence = blue_trash_can_score(
            blue_score=blue_score,
            black_score=black_score,
            aspect_ratio=aspect_ratio,
            vertical_position=vertical_position,
            contour_area=contour_area,
            min_area_px=min_area_px,
        )

        if confidence < min_confidence:
            continue

        valid_candidates.append(
            (
                confidence,
                contour,
                int(x),
                int(y),
                int(width),
                int(height),
                blue_score,
                black_score,
            )
        )

    if valid_candidates:
        valid_candidates = [max(valid_candidates, key=lambda item: item[0])]

    for confidence, _, x, y, width, height, blue_score, black_score in valid_candidates:
        detection = DetectedObject(
            class_name="trash can",
            confidence=confidence,
            x=x,
            y=y,
            width=width,
            height=height,
        )
        detection.add_attribute("color", "blue", 1.0)
        detection.add_attribute("black_band", "detected" if black_score > 0.02 else "weak", black_score)

        detections.append(detection)

        rect_contour = make_rect_contour(x, y, width, height)
        debug_overlays.append(
            DebugOverlay(
                color=overlay_color,
                contour=rect_contour,
                label="blue trash can",
                x=x,
                y=y,
            )
        )

    return detections, debug_overlays


def mask_fraction(
    mask: np.ndarray,
    x: int,
    y: int,
    width: int,
    height: int,
) -> float:
    frame_height, frame_width = mask.shape[:2]

    x0 = max(0, x)
    y0 = max(0, y)
    x1 = min(frame_width, x + width)
    y1 = min(frame_height, y + height)

    if x1 <= x0 or y1 <= y0:
        return 0.0

    crop = mask[y0:y1, x0:x1]
    return cv2.countNonZero(crop) / float(max(1, crop.size))


def marker_score(
    mask: np.ndarray,
    x: int,
    y: int,
    width: int,
    height: int,
) -> float:
    """
    Score marker visibility inside the candidate box.
    Uses the middle 10 to 90 percent of object height.
    """
    frame_height, frame_width = mask.shape[:2]

    x0 = max(0, x)
    y0 = max(0, y)
    x1 = min(frame_width, x + width)
    y1 = min(frame_height, y + height)

    if x1 <= x0 or y1 <= y0:
        return 0.0

    crop = mask[y0:y1, x0:x1]
    crop_h, crop_w = crop.shape[:2]

    if crop_w < 10 or crop_h < 10:
        return 0.0

    band_y0 = int(crop_h * 0.10)
    band_y1 = int(crop_h * 0.90)
    band_crop = crop[band_y0:band_y1, :]

    return cv2.countNonZero(band_crop) / float(max(1, band_crop.size))


def green_trash_can_score(
    green_score: float,
    red_score: float,
    black_score: float,
    aspect_ratio: float,
    vertical_position: float,
    contour_area: float,
    min_area_px: int,
) -> float:
    area_score = min(1.0, contour_area / float(max(1, min_area_px * 6)))
    color_score = min(1.0, green_score / 0.25)
    red_marker_score = min(1.0, red_score / 0.05)
    black_marker_score = min(1.0, black_score / 0.10)

    ideal_aspect = 1.26
    aspect_score = max(0.0, 1.0 - abs(aspect_ratio - ideal_aspect) / 1.8)
    position_score = 1.0 if vertical_position >= 0.08 else 0.0

    score = (
        0.25 * area_score
        + 0.25 * color_score
        + 0.25 * red_marker_score
        + 0.10 * black_marker_score
        + 0.10 * aspect_score
        + 0.05 * position_score
    )

    return max(0.0, min(1.0, score))


def blue_trash_can_score(
    blue_score: float,
    black_score: float,
    aspect_ratio: float,
    vertical_position: float,
    contour_area: float,
    min_area_px: int,
) -> float:
    area_score = min(1.0, contour_area / float(max(1, min_area_px * 6)))
    color_score = min(1.0, blue_score / 0.25)
    black_marker_score = min(1.0, black_score / 0.10)

    ideal_aspect = 1.26
    aspect_score = max(0.0, 1.0 - abs(aspect_ratio - ideal_aspect) / 1.8)
    position_score = 1.0 if vertical_position >= 0.08 else 0.0

    score = (
        0.35 * area_score
        + 0.35 * color_score
        + 0.10 * black_marker_score
        + 0.15 * aspect_score
        + 0.05 * position_score
    )

    return max(0.0, min(1.0, score))


# ---------------- Paper ball detector disabled for now ----------------

# ---------------- Paper ball detector ----------------

def detect_paper_balls(
    frame_bgr: np.ndarray,
) -> tuple[list[DetectedObject], list[DebugOverlay]]:
    """
    Detect a crumpled paper ball with pink paint/marker.

    Logic:
    - Find pink/magenta paint
    - Expand around the pink region to include the full paper ball
    - Check for visible white paper
    - Require the object to be small and roughly round/square
    - Reject large/tall trash-can-like objects
    """
    detections: list[DetectedObject] = []
    debug_overlays: list[DebugOverlay] = []

    blurred = cv2.GaussianBlur(frame_bgr, (5, 5), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    pink_mask = build_paper_ball_pink_mask(hsv)
    white_mask = build_paper_ball_white_mask(hsv)

    frame_height, frame_width = frame_bgr.shape[:2]
    frame_area = float(frame_width * frame_height)

    contours, _ = cv2.findContours(
        pink_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    valid_candidates: list[tuple[float, int, int, int, int]] = []

    for contour in contours:
        pink_area = float(cv2.contourArea(contour))

        # Ignore tiny pink noise.
        if pink_area < 20:
            continue

        px, py, pw, ph = cv2.boundingRect(contour)

        if pw < 5 or ph < 5:
            continue

        # Center of the pink painted region.
        cx = px + pw / 2.0
        cy = py + ph / 2.0

        # Expand around pink paint to include the white paper ball.
        estimated_size = int(max(pw, ph) * 3.2)
        estimated_size = max(estimated_size, 40)

        x = int(cx - estimated_size / 2.0)
        y = int(cy - estimated_size / 2.0)
        width = estimated_size
        height = estimated_size

        # Clamp to image bounds.
        x = max(0, x)
        y = max(0, y)
        width = min(width, frame_width - x)
        height = min(height, frame_height - y)

        if width <= 0 or height <= 0:
            continue

        candidate_area = float(width * height)
        object_fraction = candidate_area / frame_area

        # Reject huge objects like trash cans.
        if object_fraction > 0.12:
            continue

        # Reject very small detections.
        if width < 20 or height < 20:
            continue

        # Reject boxes that are way too big for the paper ball.
        if width > 210 or height > 210:
            continue

        aspect_ratio = width / float(max(1, height))

        # Paper ball should be roughly square/round.
        if aspect_ratio < 0.55 or aspect_ratio > 1.80:
            continue

        pink_fraction = mask_fraction_local(pink_mask, x, y, width, height)
        white_fraction = mask_fraction_local(white_mask, x, y, width, height)

        # Pink paint must be visible.
        if pink_fraction < 0.008:
            continue

        # White paper must still be visible.
        if white_fraction < 0.018:
            continue

        roundness_score = paper_ball_roundness_score(width, height)
        size_score = paper_ball_size_score(candidate_area, frame_area)
        pink_score = min(1.0, pink_fraction / 0.075)
        white_score = min(1.0, white_fraction / 0.15)

        confidence = (
            0.38 * pink_score
            + 0.30 * white_score
            + 0.22 * roundness_score
            + 0.10 * size_score
        )

        if confidence < 0.35:
            continue

        valid_candidates.append(
            (
                confidence,
                int(x),
                int(y),
                int(width),
                int(height),
            )
        )

    # Keep only the best paper ball candidate.
    if valid_candidates:
        valid_candidates = [max(valid_candidates, key=lambda item: item[0])]

    for confidence, x, y, width, height in valid_candidates:
        detection = DetectedObject(
            class_name="paper ball",
            confidence=confidence,
            x=x,
            y=y,
            width=width,
            height=height,
        )
        detection.add_attribute("marker", "pink paint", confidence)
        detection.add_attribute("color", "white/pink", confidence)

        detections.append(detection)

        rect_contour = make_paper_ball_rect_contour(x, y, width, height)

        debug_overlays.append(
            DebugOverlay(
                color=(255, 0, 255),
                contour=rect_contour,
                label=f"paper ball {confidence:.2f}",
                x=x,
                y=y,
            )
        )

    return detections, debug_overlays
def build_paper_ball_pink_mask(
    frame_hsv: np.ndarray,
) -> np.ndarray:
    """
    Pink/magenta paint mask for the paper ball.

    This targets the painted pink paper ball instead of generic red,
    which helps avoid confusion with red tape/markers on trash cans.
    """
    mask = cv2.inRange(
        frame_hsv,
        np.array((135, 45, 55), dtype=np.uint8),
        np.array((179, 255, 255), dtype=np.uint8),
    )

    open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))

    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel)

    return mask
def build_paper_ball_red_mask(
    frame_hsv: np.ndarray,
) -> np.ndarray:
    """
    Red tape mask for the paper ball.

    Red wraps around HSV hue, so this uses two ranges.
    """
    red_mask_1 = cv2.inRange(
        frame_hsv,
        np.array((0, 70, 45), dtype=np.uint8),
        np.array((12, 255, 255), dtype=np.uint8),
    )

    red_mask_2 = cv2.inRange(
        frame_hsv,
        np.array((165, 70, 45), dtype=np.uint8),
        np.array((180, 255, 255), dtype=np.uint8),
    )

    mask = cv2.bitwise_or(red_mask_1, red_mask_2)

    open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))

    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel)

    return mask


def build_paper_ball_white_mask(
    frame_hsv: np.ndarray,
) -> np.ndarray:
    """
    White/light paper mask.

    Looks for low saturation and high brightness.
    """
    mask = cv2.inRange(
        frame_hsv,
        np.array((0, 0, 105), dtype=np.uint8),
        np.array((180, 95, 255), dtype=np.uint8),
    )

    open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel)

    return mask


def mask_fraction_local(
    mask: np.ndarray,
    x: int,
    y: int,
    width: int,
    height: int,
) -> float:
    frame_height, frame_width = mask.shape[:2]

    x0 = max(0, x)
    y0 = max(0, y)
    x1 = min(frame_width, x + width)
    y1 = min(frame_height, y + height)

    if x1 <= x0 or y1 <= y0:
        return 0.0

    crop = mask[y0:y1, x0:x1]
    return cv2.countNonZero(crop) / float(max(1, crop.size))


def paper_ball_roundness_score(
    width: int,
    height: int,
) -> float:
    ratio = min(width, height) / float(max(1, max(width, height)))
    return max(0.0, min(1.0, ratio))


def paper_ball_size_score(
    candidate_area: float,
    frame_area: float,
) -> float:
    object_fraction = candidate_area / float(max(1.0, frame_area))

    # Good range for a small nearby paper ball.
    if object_fraction < 0.002:
        return 0.2

    if object_fraction > 0.12:
        return 0.0

    return 1.0


def make_paper_ball_rect_contour(
    x: int,
    y: int,
    width: int,
    height: int,
) -> np.ndarray:
    """Create a clean rectangular debug box."""
    return np.array(
        [
            [[x, y]],
            [[x + width, y]],
            [[x + width, y + height]],
            [[x, y + height]],
        ],
        dtype=np.int32,
    )