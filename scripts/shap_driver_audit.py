"""Day-1 audit (section 7 of the build plan): run SHAP on the real sample subscribers, confirm
every driver that actually appears has a src.churn.driver_to_query.DRIVER_TO_QUERY key. Run this
again any time the churn model or the sample subscriber set changes.

Run: python scripts/shap_driver_audit.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from src.churn.driver_to_query import DRIVER_TO_QUERY
from src import config
from src.churn.model_api import get_churn_risk


def main():
    msnos = pd.read_csv(config.SUBSCRIBERS_PATH)["msno"].tolist()
    top1_counts, top2_counts = {}, {}

    for msno in msnos:
        risk = get_churn_risk(msno, top_n=2)
        d0, d1 = risk.top_drivers[0].feature, risk.top_drivers[1].feature
        top1_counts[d0] = top1_counts.get(d0, 0) + 1
        top2_counts[d1] = top2_counts.get(d1, 0) + 1

    print(f"Audited {len(msnos)} sample subscribers.\n")
    print("Top-1 driver frequency:")
    for f, n in sorted(top1_counts.items(), key=lambda x: -x[1]):
        print(f"  {f:<20} {n}")
    print("\nTop-2 (secondary) driver frequency:")
    for f, n in sorted(top2_counts.items(), key=lambda x: -x[1]):
        print(f"  {f:<20} {n}")

    seen = set(top1_counts) | set(top2_counts)
    missing_keys = seen - set(DRIVER_TO_QUERY)
    print(f"\nDrivers seen in top-2 across the sample: {sorted(seen)}")
    print(f"All 12 training features have a DRIVER_TO_QUERY key: {set(DRIVER_TO_QUERY.keys()) == {
        'is_auto_renew','payment_plan_days','actual_amount_paid','plan_list_price','discount',
        'has_activity_60d','recency_days','active_days_30','secs_30','unq_30',
        'completion_ratio','activity_trend'}}")
    if missing_keys:
        print(f"[FAIL] drivers seen with NO DRIVER_TO_QUERY key: {missing_keys}")
        sys.exit(1)
    print("[PASS] every driver seen in this sample has a DRIVER_TO_QUERY key.")


if __name__ == "__main__":
    main()
