"""Finite-family risk selection (Hoeffding + union bound).

For K pre-specified configurations and n independent calibration units (recordings, not frames) with a bounded loss
in [0, 1], with probability >= 1 - delta simultaneously for all K:  R_k <= R_hat_k + sqrt(log(K / delta) / (2 n)).
Select the configuration with the best recall whose certified upper bound is <= alpha; None if none qualifies.
Assumes i.i.d. units; it does not certify physical safety or survive distribution shift.
"""
from __future__ import annotations

import math
from typing import Dict, Optional, Tuple


def hoeffding_upper(emp_risk: float, n: int, k: int, delta: float) -> float:
    return emp_risk + math.sqrt(math.log(k / delta) / (2 * n))


def select_config(configs: Dict[str, Dict[str, float]], n: int, alpha: float, delta: float = 0.05) -> Tuple[Optional[str], Dict]:
    k = len(configs)
    info = {c: {**v, "upper": hoeffding_upper(v["risk"], n, k, delta)} for c, v in configs.items()}
    ok = [c for c, v in info.items() if v["upper"] <= alpha]
    return (max(ok, key=lambda c: info[c]["recall"]) if ok else None), info
