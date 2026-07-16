"""Driver -> playbook-cluster query conditioning (section 2 of the build plan). This is the actual
differentiator: it turns the churn model's own SHAP output into a retrieval query, so the `both`
route retrieves the playbook that matches *why* this specific subscriber is at risk, instead of a
generic answer.

Audit rule: every one of the 12 training features must have a key here (verified in
scripts/shap_driver_audit.py against the real sample subscribers -- all 12 covered). If a feature
is ever added to the model, add a key here in the same commit, or the `both` path breaks silently
for that driver.
"""
DRIVER_TO_QUERY = {
    "is_auto_renew":      "auto-renew re-enablement retention for non-renewing subscribers",
    "payment_plan_days":  "plan-commitment upsell converting short-cycle subscribers to longer plans",
    "discount":           "discount-expiry retention for price-sensitive discount-dependent subscribers",
    "actual_amount_paid": "discount-expiry retention for price-sensitive discount-dependent subscribers",
    "plan_list_price":    "discount-expiry retention for price-sensitive discount-dependent subscribers",
    "has_activity_60d":   "dormant listener win-back re-engagement campaign",
    "recency_days":       "dormant listener win-back re-engagement campaign",
    "active_days_30":     "low-engagement activation building listening habits",
    "secs_30":            "low-engagement activation building listening habits",
    "unq_30":             "content discovery and catalog exploration for low-variety listeners",
    "completion_ratio":   "recommendation quality improvement for high-skip listeners",
    "activity_trend":     "declining-engagement early intervention for downward activity trend",
}


def drivers_to_query(top_drivers, n: int = 2) -> str:
    """top_drivers: list of Driver (anything with a .feature attr), already restricted to the 12
    training features by get_churn_risk. Joins the top-n DISTINCT playbook queries -- distinct
    because multiple drivers (e.g. discount + plan_list_price) can map to the same cluster, and
    sending the retriever the same query string twice wastes a retrieval slot instead of pulling in
    a second, genuinely different playbook."""
    queries = []
    for d in top_drivers:
        q = DRIVER_TO_QUERY[d.feature]
        if q not in queries:
            queries.append(q)
        if len(queries) == n:
            break
    return " ; ".join(queries)
