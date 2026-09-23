"""Shared vocab and helpers for easy-example generators."""
import random

LINES = ["SMT-L1", "SMT-L2", "SMT-L3", "SMT-L4", "SMT-L5", "SMT-L6"]
PRINTERS = ["DEK-01", "DEK-02", "DEK-03"]
MOUNTERS = ["NXT-03", "NXT-05", "SIPLACE-02", "SIPLACE-04", "M-05", "M-07", "YSM-04", "YSM-06"]
REFLOWS = ["R-01", "R-02", "R-03"]
AOIS = ["AOI-01", "AOI-02", "爐後 AOI-03"]
SPIS = ["SPI-01", "SPI-02"]
TESTERS = ["ICT-02", "FCT-01", "X-ray-01"]
PARTS = ["0402 電阻", "0603 電容", "0805 電阻", "QFN-48", "BGA-256", "連接器 CON-12", "SOT-23 電晶體", "LED 0603"]
PNS = ["R123-0402-10K", "C220-0603-1U", "U301-QFN48", "U105-BGA256", "J12-CON-2x10", "Q7-SOT23", "L4-0805-2R2"]
NAMES = ["小陳", "阿明", "王工", "李組長", "Amy", "阿華", "小林", "張工"]


def wo(rng: random.Random) -> str:
    return f"WO-202609{rng.randint(10, 28):02d}-{rng.randint(1, 199):03d}"


def hhmm(rng: random.Random) -> str:
    return f"{rng.randint(0, 23):02d}:{rng.choice([0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55]):02d}"


def pick(rng: random.Random, xs):
    return rng.choice(xs)


def slot(rng: random.Random) -> int:
    return rng.randint(1, 60)
