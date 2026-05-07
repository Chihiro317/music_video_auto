from __future__ import annotations

import math


def beat_scale(t: float, beats: list[float], base: float = 1.0, pulse: float = 0.08, decay: float = 0.18) -> float:
    """Return a scale factor that pulses after nearby beats."""
    if not beats:
        return base + 0.015 * math.sin(t * 2.0)
    last = None
    for beat in beats:
        if beat <= t:
            last = beat
        else:
            break
    if last is None:
        return base
    delta = t - last
    if delta > decay:
        return base + 0.015 * math.sin(t * 2.0)
    strength = 1.0 - delta / decay
    return base + pulse * strength


def shake_offset(t: float, beats: list[float], amount: int = 10) -> tuple[int, int]:
    """Return small x/y shake offsets."""
    near = False
    for beat in beats:
        if 0 <= t - beat <= 0.16:
            near = True
            break
    if not near:
        return 0, int(4 * math.sin(t * 3.1))
    x = int(amount * math.sin(t * 75))
    y = int(amount * 0.6 * math.cos(t * 90))
    return x, y


def flash_opacity(t: float, beats: list[float], duration: float = 0.08) -> float:
    """Return white flash opacity after beat."""
    for beat in beats:
        delta = t - beat
        if 0 <= delta <= duration:
            return 0.28 * (1.0 - delta / duration)
    return 0.0
