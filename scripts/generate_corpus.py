"""Generate the synthetic document corpus (section 5 of the build plan).

Docs are honestly synthetic (no real KKBox retention playbooks exist), but heterogeneous in
structure and vocabulary so BM25 and dense retrieval have something real to differentiate, and
so each of the 12 SHAP driver features has language it can actually retrieve on.

Run: python scripts/generate_corpus.py  -> writes data/raw/corpus/*.pdf and *.csv
"""
import csv
import sys
import textwrap
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import fitz  # PyMuPDF

from src import config

REPO_ROOT = config.REPO_ROOT
OUT_DIR = config.CORPUS_DIR
OUT_DIR.mkdir(parents=True, exist_ok=True)

PAGE_W, PAGE_H = 595, 842   # A4 points
MARGIN = 46
FONTSIZE = 10.5
LINE_H = FONTSIZE + 3.4
CHARS_PER_LINE = 92


def make_pdf(path: Path, title: str, sections: list[tuple[str, str]]):
    """sections: list of (heading, body). body paragraphs separated by \\n\\n."""
    doc = fitz.open()
    lines = [("title", title), ("", "")]
    for heading, body in sections:
        lines.append(("heading", heading))
        for para in body.strip().split("\n\n"):
            para = " ".join(para.split())  # collapse internal whitespace/newlines
            for wrapped in textwrap.wrap(para, CHARS_PER_LINE) or [""]:
                lines.append(("body", wrapped))
            lines.append(("", ""))

    page = None
    y = MARGIN
    for kind, text in lines:
        if page is None or y > PAGE_H - MARGIN:
            page = doc.new_page(width=PAGE_W, height=PAGE_H)
            y = MARGIN
        if kind == "title":
            page.insert_text((MARGIN, y), text, fontsize=16, fontname="helv")
            y += 24
        elif kind == "heading":
            y += 6
            page.insert_text((MARGIN, y), text, fontsize=12.5, fontname="helv")
            y += 18
        else:
            page.insert_text((MARGIN, y), text, fontsize=FONTSIZE, fontname="helv")
            y += LINE_H
    doc.save(path)
    doc.close()
    print(f"wrote {path.relative_to(REPO_ROOT)} ({len(doc)} internal calls)" if False else f"wrote {path.relative_to(REPO_ROOT)}")


# ---------------------------------------------------------------------------
# Retention playbooks P1-P6 (both, policy routes)  -  each aligned to one driver cluster
# ---------------------------------------------------------------------------

PLAYBOOKS = {
    "playbook_p1_auto_renew_reenablement.pdf": (
        "Retention Playbook P1  -  Auto-Renew Re-enablement",
        [
            ("When this applies",
             """
             This playbook targets subscribers whose auto-renew flag is off (is_auto_renew = 0).
             This is the single strongest churn driver in the scoring model: a subscriber who has
             turned off auto-renewal will lapse silently at the end of their current billing cycle
             unless they take a manual action to re-subscribe. Unlike engagement-based risk, this
             is a structural risk  -  the subscriber may still be actively listening, but the account
             is on a countdown to non-renewal regardless of usage.

             Auto-renew-off subscribers should be treated as the highest-priority contact segment.
             In the current scoring population, this single flag accounts for the majority of the
             SHAP-attributed churn risk across the paying base.
             """),
            ("Root causes to distinguish before choosing an offer",
             """
             Auto-renew can be off for several different reasons, and the right playbook branch
             depends on which one applies:
             (1) Accidental or forgotten opt-out  -  the subscriber disabled auto-renew during a
             promo period or after a billing dispute and never turned it back on.
             (2) Active price sensitivity  -  the subscriber disabled auto-renew specifically to
             avoid being charged full price when a discount period ends (see Playbook P3 for the
             price-sensitive branch; these two often co-occur and should be handled together).
             (3) Deliberate churn intent  -  the subscriber has already decided to leave and turned
             off auto-renew as the first step; engagement signals (recency, activity_trend) help
             distinguish this group from the merely-forgetful group.
             """),
            ("Recommended interventions",
             """
             Tier 1 (7 days before cycle end): send a renewal reminder that surfaces the exact
             renewal date and price, with a single-tap re-enable action. No discount is offered at
             this stage  -  most reactivations at this tier are the "forgotten opt-out" group and do
             not require a financial incentive.

             Tier 2 (2 days before cycle end, no action taken): escalate to a retention offer.
             If actual_amount_paid is already well below plan_list_price (an active discount), do
             not add a further discount  -  instead offer a plan-length concession (e.g. switch to a
             shorter payment_plan_days cycle at the same rate) to reduce the size of the commitment
             the subscriber is being asked to re-enable.

             Tier 3 (cycle has lapsed): treat as a win-back case, not a renewal case  -  hand off to
             Playbook P4 (dormant win-back) rather than continuing to push auto-renew messaging,
             since the account is now inactive rather than merely non-renewing.
             """),
            ("What NOT to do",
             """
             Do not lead with a discount for subscribers who re-enabled auto-renew on their own in
             the last 90 days  -  this trains the base to churn strategically in order to farm offers.
             Do not conflate this playbook with dormant win-back (P4): a subscriber with auto-renew
             off but high recent activity is a payment-mechanics problem, not an engagement problem,
             and a discovery-focused campaign will not move them.
             """),
        ],
    ),
    "playbook_p2_plan_commitment_upsell.pdf": (
        "Retention Playbook P2  -  Plan-Commitment Upsell",
        [
            ("When this applies",
             """
             This playbook targets subscribers on short payment_plan_days cycles (for example,
             month-to-month or 30-day plans) as opposed to longer commitment plans. Short-cycle
             subscribers churn at a materially higher rate than subscribers on longer plans, because
             every cycle boundary is a fresh decision point to leave, whereas a longer plan removes
             several of those decision points entirely.

             This is a commitment-structure risk, distinct from engagement or price risk: a
             short-cycle subscriber can be highly engaged and still churn simply because they are
             asked to re-decide every month.
             """),
            ("Recommended interventions",
             """
             Offer a plan-length upgrade (e.g. move a 30-day plan to a 90-day or annual plan) at
             renewal time, framed around convenience ("stop worrying about renewing every month")
             rather than around price. A modest per-cycle discount (5-10%) tied specifically to the
             longer commitment is appropriate here and should be framed as a loyalty rate, not a
             churn-prevention discount  -  it is being paid for by locking in the commitment, not by
             the subscriber's risk level.

             For subscribers who decline the upgrade twice, do not push a third time in the same
             quarter; over-offering a plan-length upsell to a subscriber who has explicitly signalled
             a preference for short commitment increases the chance they cancel outright rather than
             just declining the upsell.
             """),
            ("Interaction with other drivers",
             """
             Plan-commitment risk frequently co-occurs with auto-renew risk (P1), since a short
             cycle simply gives auto-renew more chances to be tested. When both are present, lead
             with the plan-length upsell first  -  a longer plan structurally reduces how often the
             auto-renew risk can even manifest, whereas fixing auto-renew alone leaves the short-cycle
             exposure in place for the next month.
             """),
        ],
    ),
    "playbook_p3_discount_expiry_price_sensitivity.pdf": (
        "Retention Playbook P3  -  Discount-Expiry & Price-Sensitivity Retention",
        [
            ("When this applies",
             """
             This playbook targets subscribers who are discount-dependent: actual_amount_paid is
             substantially below plan_list_price, or the discount field is high relative to the plan
             tier. This is a validated, real churn driver, not a collinearity artifact  -  raw churn
             rates rise sharply across price quartiles even after controlling for the discounted
             segment, meaning the price effect holds up on its own and is not just riding on
             auto-renew status.

             The core risk moment is the discount's expiry date: subscribers in this segment churn
             disproportionately in the cycle where their promotional rate rolls off to full
             plan_list_price, not randomly across the subscription lifecycle.
             """),
            ("Recommended interventions",
             """
             Do not simply re-offer the same discount indefinitely  -  this teaches the base that
             cancelling (or threatening to) is the way to extract a renewed discount, and it does
             not address the underlying price sensitivity, it just delays the same churn event to
             the next expiry.

             Instead, step the discount down gradually (e.g. 40% off the first term, 25% the
             second, 10% the third) so the price gap the subscriber has to absorb at any single
             renewal is smaller, and frame the communication around what the subscriber has been
             using (song variety, listening history) rather than around the price change itself.

             For subscribers on the highest list-price tiers, consider a plan-tier downgrade offer
             instead of a discount on the current tier  -  the subscriber keeps paying full price for
             a plan they can afford, which does not carry the same "waiting for the discount to
             expire again" dynamic as a discount renewal.
             """),
            ("Segment note",
             """
             Price sensitivity is concentrated in specific plan tiers (plan_list_price extremes) and
             should not be treated as a uniform population  -  a small discount matters much less to a
             subscriber on a premium multi-device plan than to a subscriber on the cheapest single
             device tier. Segment the offer size to the list price tier, not to a flat percentage.
             """),
        ],
    ),
    "playbook_p4_dormant_winback.pdf": (
        "Retention Playbook P4  -  Dormant Listener Win-Back",
        [
            ("When this applies",
             """
             This playbook targets subscribers who are still paying but are no longer really using
             the service: has_activity_60d = 0 (no listening activity in the last 60 days) or a high
             recency_days value (a long gap since the last session). This is a "paying but gone"
             segment  -  the churn risk is not about price or plan mechanics, it is that the
             subscriber has simply stopped getting value from the product and the cancellation is
             just a matter of time until they notice the charge.

             Recency is one of the cleanest signals in the model: subscribers with a long gap since
             last activity churn at a much higher rate than active listeners, cleanly and
             monotonically with the size of the gap.
             """),
            ("Recommended interventions",
             """
             Lead with content, not price. A dormant subscriber does not need a discount, they need
             a reason to open the app again. Send a small number (2-3) of highly personalized
             re-engagement prompts built from their pre-dormancy listening history ("new releases
             from artists you used to play") rather than a generic "come back" push notification.

             If two re-engagement prompts over consecutive weeks produce no return session, stop the
             content-led approach and escalate to a win-back offer (a free extended trial period or
             a one-time discount) framed as "we miss you" rather than "renew now"  -  the goal at this
             point is to get one more listening session, not to defend the renewal directly.

             Do not run a price-led win-back campaign as the first touch for this segment  -  it treats
             a product-fit / relevance problem as a price problem and tends to convert poorly, because
             a subscriber who has stopped listening was not going to be retained by a lower price on a
             service they were not using anyway.
             """),
            ("Distinguishing from low-engagement (P5)",
             """
             Dormant (this playbook) means essentially zero activity  -  the subscriber has
             functionally left already. Low-engagement (Playbook P5) means the subscriber is still
             using the service, just thinly (fewer active days, lower listening volume). Sending the
             P4 "we miss you" win-back message to a P5 subscriber who is still checking in a few times
             a month reads as tone-deaf and can itself prompt a cancellation; check has_activity_60d
             and recency_days before choosing between the two playbooks.
             """),
        ],
    ),
    "playbook_p5_low_engagement_activation.pdf": (
        "Retention Playbook P5  -  Low-Engagement Activation & Discovery",
        [
            ("When this applies",
             """
             This playbook targets subscribers who are still active but thinly engaged: low
             active_days_30 (few distinct days used in the last month), low secs_30 (low total
             listening volume), and/or low unq_30 (a narrow range of unique tracks  -  the subscriber
             is replaying a small, unchanging set rather than exploring the catalog). This is
             distinct from dormancy (Playbook P4): the account is genuinely in use, but the
             engagement is thin enough that it is at risk of tipping into dormancy.

             A narrowing unq_30 in particular signals a subscriber settling into a small comfort
             zone of tracks, which frequently precedes a drop in active_days_30 the following month
             as the experience becomes repetitive.
             """),
            ("Recommended interventions",
             """
             For low active_days_30 / low secs_30: build a listening habit rather than pushing a
             single large content recommendation. A short daily or every-other-day "quick pick"
             notification at a consistent time of day (matched to the subscriber's historical
             listening hours where available) is more effective at growing active_days_30 than an
             infrequent large content drop, because it targets the frequency metric directly.

             For low unq_30 (narrow catalog use): the intervention is discovery-oriented rather than
             frequency-oriented. Surface adjacent artists and playlists near the subscriber's existing
             comfort zone rather than a completely novel genre  -  a large jump in recommended content
             away from established preferences is more likely to be ignored than a moderate expansion
             just beyond it.

             These two branches (frequency vs. discovery) should not be bundled into a single generic
             "use the app more" message  -  a subscriber who listens narrowly but daily needs catalog
             discovery, while a subscriber who listens broadly but rarely needs a frequency nudge, and
             sending the wrong one wastes the contact.
             """),
        ],
    ),
    "playbook_p6_recommendation_quality.pdf": (
        "Retention Playbook P6  -  Recommendation Quality & Early Intervention",
        [
            ("When this applies",
             """
             This playbook has two related triggers. First, a low completion_ratio (the subscriber
             skips a high proportion of tracks rather than finishing them)  -  this signals
             recommendation dissatisfaction: the subscriber is opening the app and starting sessions,
             but what is being served is not landing, which is a different failure mode from simply
             not using the app. Second, a negative activity_trend (listening activity is declining
             relative to the prior period)  -  this is an early-warning signal that should be acted on
             before it shows up as low active_days_30 or dormancy.
             """),
            ("Recommended interventions  -  completion_ratio",
             """
             A high skip rate is a signal to intervene on the recommendation experience itself, not
             on the subscriber's usage habits. Prompt the subscriber to explicitly re-rate or refresh
             their taste profile (a short "tell us what you like" flow) rather than sending more
             volume of the same kind of recommendation that is already being skipped. Where available,
             surface a small number of curated, editorially-selected playlists as an alternative to the
             algorithmic feed for a trial period  -  this isolates whether the issue is the
             recommendation model or a genuine mismatch between the subscriber's taste and the
             catalog.
             """),
            ("Recommended interventions  -  activity_trend",
             """
             A negative activity_trend should be treated as the earliest actionable warning in the
             engagement family  -  by the time active_days_30 or recency_days themselves look bad, the
             subscriber is already most of the way to dormant. Because this is a leading indicator
             rather than a confirmed problem, the appropriate response is lightweight: a single
             check-in style prompt surfacing something new and relevant, without the heavier
             escalation used for confirmed dormancy (Playbook P4). Treat a negative trend as a
             one-touch nudge, not a full win-back campaign  -  over-escalating on a leading indicator
             risks annoying subscribers who would have recovered on their own.
             """),
        ],
    ),
}

for fname, (title, sections) in PLAYBOOKS.items():
    make_pdf(OUT_DIR / fname, title, sections)


# ---------------------------------------------------------------------------
# Subscription & renewal policy (policy route)
# ---------------------------------------------------------------------------

make_pdf(
    OUT_DIR / "policy_subscription_renewal.pdf",
    "Subscription & Renewal Policy",
    [
        ("Auto-renew mechanics",
         """
         Every paid subscription has an auto-renew setting, on or off, controlled by the subscriber
         at any time from the account page. When auto-renew is on, the subscription charges the
         plan's list price (less any active discount) automatically at the end of the current
         payment_plan_days cycle and the cycle continues uninterrupted. When auto-renew is off, the
         subscription reaches its expiry date at the end of the current cycle and access lapses
         unless the subscriber manually renews before that date.

         Turning auto-renew off does not cancel the current cycle  -  the subscriber retains access
         until the existing paid period ends. It only prevents the automatic charge that would start
         the next cycle.
         """),
        ("Plan lengths and pricing",
         """
         Plans are offered at several payment_plan_days lengths (commonly 30, 90, 180, and 365 days).
         Longer plans carry a lower effective per-day rate (plan_list_price divided across more days)
         in exchange for the longer up-front commitment. A subscriber can change plan length at any
         renewal boundary; a mid-cycle plan change takes effect at the next renewal, not immediately,
         to avoid pro-rating disputes.
         """),
        ("Cancellation",
         """
         A subscriber may cancel at any time. Cancellation before the end of a paid cycle does not
         issue a refund for the remaining days by default; access continues through the end of the
         period already paid for, after which the account lapses. A cancellation is recorded
         separately from a simple auto-renew-off, since a cancellation also blocks any pending
         renewal transaction outright rather than merely not scheduling a new one.
         """),
        ("Discounts and promotional pricing",
         """
         A discount reduces actual_amount_paid below the plan's plan_list_price for a defined
         promotional window. Unless a subscriber is on an explicitly recurring loyalty rate, a
         discount applies to a fixed number of cycles and the plan reverts to plan_list_price at the
         discount's expiry, at which point the subscriber will see a higher charge on the next
         auto-renewal unless a new offer is applied before that date.
         """),
    ],
)


# ---------------------------------------------------------------------------
# Campaign manual (policy route)
# ---------------------------------------------------------------------------

make_pdf(
    OUT_DIR / "campaign_manual.pdf",
    "Retention Campaign Operations Manual",
    [
        ("Campaign cadence",
         """
         Retention campaigns run on a weekly contact cycle. A subscriber flagged for outreach in a
         given week is not re-contacted for the same underlying reason for at least 21 days, to avoid
         message fatigue. An exception applies to the tiered auto-renew reminder sequence (Playbook
         P1), which runs on its own fixed 7-day / 2-day schedule ahead of a known renewal date rather
         than the general 21-day cooldown.
         """),
        ("Channel selection",
         """
         In-app notifications are the default channel for engagement-oriented playbooks (dormant
         win-back, low-engagement activation, recommendation quality), since these interventions rely
         on the subscriber opening the app to experience the fix. Email is the default channel for
         billing-oriented playbooks (auto-renew reminders, discount-expiry notices, plan-commitment
         offers), since these need to be actionable even if the subscriber has not opened the app
         recently.
         """),
        ("Offer governance",
         """
         Any discount-bearing offer must be logged against the subscriber's account with an expiry
         date and a reason code tied to the playbook that generated it. A subscriber may not be on
         more than one active discount-bearing offer at a time; if a new playbook would generate a
         second concurrent offer, the higher-value offer wins and the other is suppressed rather than
         stacked.
         """),
        ("Escalation and hand-off between playbooks",
         """
         Campaigns are expected to hand off between playbooks as a subscriber's signals change rather
         than running independently. The two required hand-offs are: (1) an auto-renew reminder
         sequence (P1) that reaches its final tier without a renewal hands off to dormant win-back
         (P4) rather than continuing to send renewal-framed messages to a now-lapsed account; (2) a
         negative activity_trend nudge (P6) that does not arrest the decline within one contact cycle
         escalates to the full dormant win-back sequence (P4) rather than repeating the same
         lightweight nudge indefinitely.
         """),
    ],
)


# ---------------------------------------------------------------------------
# Customer support FAQ (policy route)  -  Q&A structure, deliberately different shape
# ---------------------------------------------------------------------------

make_pdf(
    OUT_DIR / "support_faq.pdf",
    "Customer Support FAQ",
    [
        ("Q: Why was I charged after I thought I cancelled?",
         """
         Turning off auto-renew and cancelling are different actions. If auto-renew was turned off,
         the subscription still runs through the end of the already-paid cycle and no further charge
         should occur after that date. If a charge occurred after an explicit cancellation, that is a
         billing error and should be escalated for a refund review, not treated as a policy question.
         """),
        ("Q: My discount disappeared and my price went up. Is that a mistake?",
         """
         Discounts apply for a fixed promotional window and revert to the plan's full list price at
         expiry; this is expected behavior, not an error. Subscribers can check the account page for
         the exact expiry date of any active discount before it rolls off, and support can proactively
         offer a renewed promotion if one is available for that subscriber's segment.
         """),
        ("Q: Can I switch to a longer or shorter plan?",
         """
         Yes. Plan length changes take effect at the next renewal boundary rather than immediately,
         to avoid a mid-cycle pro-rating dispute. A subscriber wanting an immediate change should be
         informed of the effective date clearly so they are not surprised by a charge under the old
         plan terms before the change lands.
         """),
        ("Q: The app keeps recommending songs I don't like. Can that be fixed?",
         """
         Yes  -  recommendations improve as the subscriber skips or favorites more tracks, and a
         subscriber can manually reset their taste profile from the account settings if the feed feels
         persistently off. Support should offer this reset as a concrete action rather than a generic
         "recommendations improve over time" response, since a high skip rate is a real signal worth
         acting on immediately rather than waiting out.
         """),
        ("Q: I haven't used the app in months but I'm still being charged. What are my options?",
         """
         The subscriber can cancel at any time from the account page, effective at the end of the
         current paid cycle. If the subscriber intends to keep the account but simply has not had time
         to use it, support can mention that a short break in listening does not affect their
         library, playlists, or account history when they do come back.
         """),
    ],
)


# ---------------------------------------------------------------------------
# Historical retention notes (both route)  -  case-note style, short and terse on purpose
# ---------------------------------------------------------------------------

make_pdf(
    OUT_DIR / "historical_retention_notes.pdf",
    "Historical Retention Case Notes",
    [
        ("Case 2024-0113",
         """
         Subscriber flagged with auto-renew off and no listening activity in 60+ days (dormant +
         non-renewing). Content-led win-back message alone did not produce a return session over two
         weeks. Escalated to a combined offer: one month free plus a renewal reminder. Subscriber
         returned and re-enabled auto-renew. Note: the dormant signal here was the dominant blocker  - 
         the auto-renew reminder alone (without addressing dormancy first) had previously failed on a
         similar profile earlier the same quarter.
         """),
        ("Case 2024-0187",
         """
         Subscriber on a heavily discounted plan (actual_amount_paid roughly 40% of plan_list_price)
         cancelled in the exact cycle the discount expired, consistent with the discount-expiry
         pattern. A stepped-down discount (25% instead of the original 40%, rather than a flat
         re-offer of the original rate) was tested on a similar profile the following month and
         retained the subscriber at a smaller ongoing discount rather than the full original rate.
         """),
        ("Case 2024-0249",
         """
         Subscriber showed a negative activity_trend for three consecutive weeks with active_days_30
         still in a normal range  -  engagement was falling but not yet low enough to look dormant. A
         single lightweight recommendation refresh prompt was sent (Playbook P6 style, not a full
         win-back). Activity_trend recovered to flat the following month without further contact,
         suggesting the early, low-intensity touch was sufficient and a heavier win-back campaign
         would have been unnecessary spend.
         """),
        ("Case 2024-0301",
         """
         Subscriber on a 30-day plan with high completion_ratio and good active_days_30 (clearly
         engaged) still churned at a renewal boundary. Post-hoc review found no dormancy or price
         signal  -  the account had simply never been offered a longer plan and cancelled after a
         short lapse in remembering to renew manually. This case motivated adding the plan-commitment
         upsell (Playbook P2) as a proactive offer at the first renewal for new short-cycle
         subscribers, rather than waiting for an at-risk signal to appear.
         """),
    ],
)


# ---------------------------------------------------------------------------
# Offer-eligibility rules (policy, both routes)  -  tabular, pandas-ingested
# ---------------------------------------------------------------------------

OUT_DIR.mkdir(parents=True, exist_ok=True)
with open(OUT_DIR / "offer_eligibility_rules.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["offer_id", "offer_name", "eligibility_condition", "discount_pct",
                "max_duration_days", "driver_tag", "notes"])
    rows = [
        ["OFR-101", "Renewal Reminder (no discount)", "is_auto_renew = 0, within 7 days of expiry",
         0, 0, "is_auto_renew",
         "No financial incentive; targets the forgotten-opt-out segment of Playbook P1."],
        ["OFR-102", "Plan-Length Concession", "is_auto_renew = 0, active discount already applied",
         0, 90, "is_auto_renew",
         "Shorten the ask (shorter plan) instead of stacking a second discount."],
        ["OFR-201", "Plan-Commitment Loyalty Rate", "payment_plan_days <= 30, tenure_days >= 60",
         10, 365, "payment_plan_days",
         "Requires upgrading to a 90-day plan or longer; framed as loyalty, not churn-prevention."],
        ["OFR-301", "Stepped Discount Renewal", "discount active, within 14 days of discount expiry",
         25, 90, "discount",
         "Step down from the prior discount pct; never re-offer the original rate flat."],
        ["OFR-302", "Plan-Tier Downgrade", "plan_list_price in top tier, actual_amount_paid discounted",
         0, 365, "plan_list_price",
         "Offer a lower list-price tier at full price instead of a discount on the current tier."],
        ["OFR-401", "Dormant Win-Back Free Month", "has_activity_60d = 0 or recency_days >= 60",
         100, 30, "has_activity_60d",
         "Only after 2 content-led re-engagement prompts have failed to produce a session."],
        ["OFR-501", "Habit-Building Quick Pick", "active_days_30 < 8 or secs_30 in bottom quartile",
         0, 0, "active_days_30",
         "No discount; a scheduled notification, not a billing offer."],
        ["OFR-502", "Catalog Discovery Push", "unq_30 in bottom quartile",
         0, 0, "unq_30",
         "Adjacent-artist recommendations; no discount attached."],
        ["OFR-601", "Taste Refresh Prompt", "completion_ratio in bottom quartile",
         0, 0, "completion_ratio",
         "In-app taste re-rating flow, not a discount; isolates model vs. catalog mismatch."],
        ["OFR-602", "Early Trend Check-In", "activity_trend < 0 for 2+ consecutive periods",
         0, 0, "activity_trend",
         "Single lightweight touch; escalate to OFR-401 if trend does not recover in one cycle."],
    ]
    w.writerows(rows)
print(f"wrote {(OUT_DIR / 'offer_eligibility_rules.csv').relative_to(REPO_ROOT)}")

print(f"\ncorpus complete: {len(list(OUT_DIR.glob('*')))} files in {OUT_DIR.relative_to(REPO_ROOT)}")
