# app/search/filters.py
# 슬롯 기반 1차 필터링
from __future__ import annotations
from typing import Mapping, Any
import pandas as pd


def filter_by_slots(df: pd.DataFrame, slots: Mapping[str, Any]) -> pd.DataFrame:
    """
    slots을 이용해 1차 후포 필터링
    """
    if df.empty:
        return df

    base_text = (
        (
            df.get("title", "")
            + " "
            + df.get("features_text", "")
            + " "
            + df.get("description_text", "")
            + " "
            + df.get("categories_text", "")
        )
        .fillna("")
        .astype(str)
    )
    # 혹시 모를 대소문자 섞임 방지
    base_text = base_text.str.lower()

    mask = pd.Series(True, index=df.index)

    def apply_slot(slot_key: str):
        nonlocal mask
        values = slots.get(slot_key) or []
        values = [str(v).strip().lower() for v in values if str(v).strip()]
        if not values:
            return
        slot_mask = pd.Series(False, index=df.index)
        for v in values:
            slot_mask |= base_text.str.contains(v)
        mask &= slot_mask

    for key in ["color", "material", "garment", "season", "style", "fit", "length"]:
        apply_slot(key)
    return df[mask]
