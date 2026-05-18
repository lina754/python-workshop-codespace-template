"""
Generates data/claims_sample.csv — 20 000 rows of synthetic insurance claims.

Intentional messiness introduced:
  - ~3 % missing claim_amount
  - ~1 % peril = "UNKNOWN" or empty string
  - ~0.5 % claim_date in DD/MM/YYYY format instead of ISO 8601
  - A handful of duplicate policy_ids (same policy, different claim date)

Run from the repo root:
    python data/generate_claims.py
"""

import random
import string
import numpy as np
import pandas as pd

SEED = 42
N = 20_000
OUT = "data/claims_sample.csv"

rng = np.random.default_rng(SEED)
random.seed(SEED)

# ── policy IDs ────────────────────────────────────────────────────────────────
policy_ids = [f"POL-{''.join(random.choices(string.digits, k=6))}" for _ in range(N)]

# Introduce ~30 duplicate policy IDs so the same policy appears with different
# claim dates (realistic: multiple claims on one policy over the years).
dup_sources = random.sample(range(N), 30)
for i, src in enumerate(dup_sources):
    policy_ids[random.randint(0, N - 1)] = policy_ids[src]

# ── claim dates ───────────────────────────────────────────────────────────────
start = np.datetime64("2020-01-01")
end   = np.datetime64("2024-12-31")
days_range = (end - start).astype(int) + 1
offsets = rng.integers(0, days_range, size=N)
claim_dates = start + offsets.astype("timedelta64[D]")
claim_dates_str = [str(d) for d in claim_dates]  # ISO by default

# ~0.5 % → DD/MM/YYYY
dmask = rng.random(N) < 0.005
for i in np.where(dmask)[0]:
    d = claim_dates[i].astype("datetime64[D]").astype(object)
    claim_dates_str[i] = f"{d.day:02d}/{d.month:02d}/{d.year}"

# ── perils ────────────────────────────────────────────────────────────────────
valid_perils = ["fire", "flood", "theft", "liability", "storm", "water_damage"]
peril_weights = [0.12, 0.18, 0.20, 0.15, 0.22, 0.13]
perils = rng.choice(valid_perils, size=N, p=peril_weights)

# ~1 % → "UNKNOWN" or ""
pmask = rng.random(N) < 0.01
for i in np.where(pmask)[0]:
    perils[i] = rng.choice(["UNKNOWN", ""])

# ── regions ───────────────────────────────────────────────────────────────────
regions = ["BE-BRU", "BE-VLG", "BE-WAL", "NL-NH", "NL-ZH", "DE-BAY", "DE-NRW", "FR-IDF"]
region_weights = [0.10, 0.18, 0.12, 0.13, 0.12, 0.11, 0.14, 0.10]
row_regions = rng.choice(regions, size=N, p=region_weights)

# ── claim amounts (log-normal, ~500–500 000) ──────────────────────────────────
# ln(500)≈6.2  ln(500000)≈13.1  → mu≈9.5, sigma≈1.2
raw_amounts = rng.lognormal(mean=9.5, sigma=1.2, size=N)
raw_amounts = np.clip(raw_amounts, 500, 500_000)
claim_amounts = np.round(raw_amounts, 2).astype(object)

# ~3 % → NaN
amask = rng.random(N) < 0.03
for i in np.where(amask)[0]:
    claim_amounts[i] = np.nan

# ── premiums (correlated with amount but noisy, 200–5 000) ───────────────────
log_amounts = np.where(amask, np.nanmean(np.log(raw_amounts)), np.log(raw_amounts))
log_prem = 0.55 * log_amounts + rng.normal(0, 0.8, size=N) + 2.5
raw_prem = np.exp(log_prem)
premiums = np.round(np.clip(raw_prem, 200, 5_000), 2)

# ── loss year (redundant by design) ──────────────────────────────────────────
# Parse ISO or DD/MM/YYYY to extract year
def _year(s):
    if "/" in s:
        return int(s.split("/")[2])
    return int(s[:4])

loss_years = [_year(d) for d in claim_dates_str]

# ── assemble & write ──────────────────────────────────────────────────────────
df = pd.DataFrame({
    "policy_id":    policy_ids,
    "claim_date":   claim_dates_str,
    "peril":        perils,
    "region":       row_regions,
    "claim_amount": claim_amounts,
    "premium":      premiums,
    "loss_year":    loss_years,
})

df.to_csv(OUT, index=False)
print(f"Written {len(df):,} rows → {OUT}")
print(f"  missing claim_amount : {df['claim_amount'].isna().sum():,}")
print(f"  bad perils           : {(df['peril'].isin(['UNKNOWN', ''])).sum():,}")
print(f"  DD/MM/YYYY dates     : {dmask.sum():,}")
print(f"  duplicate policy_ids : {df['policy_id'].duplicated().sum():,}")
