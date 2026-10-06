"""How numbers and long strings are written, shared by every renderer."""

MINUS = "−"


def clip(text, width: int) -> str:
    """`text` cut to `width` characters, the cut marked with an ellipsis."""
    text = str(text)
    return text if len(text) <= width else text[: width - 1] + "…"


def num(v: float) -> str:
    """A model-output quantity at about four significant digits."""
    a = abs(v)
    if a >= 1000:
        text = f"{a:,.0f}"
    elif a >= 100:
        text = f"{a:.1f}"
    elif a >= 1:
        text = f"{a:.2f}"
    else:
        text = f"{a:.3f}"
    negative = v < 0 and float(text.replace(",", "")) != 0
    return (MINUS if negative else "") + text


def decimals(values) -> int:
    """Decimals that show a column of quantities at about four significant
    digits of its largest entry — one format for the whole column."""
    top = max((abs(v) for v in values), default=0.0)
    if top >= 1000:
        return 0
    if top >= 100:
        return 1
    return 2 if top >= 1 else 3


def fixed(v: float, places: int, sign: bool = False) -> str:
    """`v` with `places` decimals, a real minus sign, and `+` if `sign`."""
    text = f"{abs(v):,.{places}f}"
    if v < 0 and float(text.replace(",", "")) != 0:
        return MINUS + text
    return ("+" if sign else "") + text


def signed(v: float) -> str:
    """`num` with an explicit sign — a contribution."""
    text = num(v)
    return text if text.startswith(MINUS) else "+" + text


def plural(n: int, word: str) -> str:
    return f"{n:,} {word}" + ("" if n == 1 else "s")
