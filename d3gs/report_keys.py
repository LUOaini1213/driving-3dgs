"""Stable keys shared by numeric report producers and readers."""
import math


def offset_key(value):
    """Keep round-trip precision for sub-metre offsets and historical integer labels."""
    value = float(value)
    if not math.isfinite(value):
        raise ValueError("offset must be finite")
    return f"{int(value):+d}" if value.is_integer() else ("+" if value >= 0 else "") + repr(value)
