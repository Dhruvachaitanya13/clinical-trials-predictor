"""
Day 5 helper: prints the exact numbers you need to fill the dashboard 'findings'.
Run from the project root:  python dashboard/findings_numbers.py
Then copy each number into the matching ___ blank in dashboard/app.py.
"""
from pathlib import Path
import pandas as pd

DATA = Path(__file__).resolve().parents[1] / "data" / "trials_features.csv"
df = pd.read_csv(DATA)
df["completed"] = (df["overall_status"].str.upper() == "COMPLETED").astype(int)
df["phase"] = df["phase"].fillna("UNKNOWN").str.upper()
df["sponsor_class"] = df["sponsor_class"].fillna("UNKNOWN")

pct = lambda x: f"{x*100:.0f}%"

print("\n=== 1. COMPLETION BY PHASE ===")
by_phase = df.groupby("phase")["completed"].mean().sort_values(ascending=False)
print(by_phase.map(pct).to_string())
print(f"Highest: {by_phase.index[0]} ({pct(by_phase.iloc[0])})  |  "
      f"Lowest: {by_phase.index[-1]} ({pct(by_phase.iloc[-1])})")

print("\n=== 2. COMPLETION BY SPONSOR ===")
by_sp = df.groupby("sponsor_class")["completed"].mean().sort_values(ascending=False)
print(by_sp.map(pct).to_string())

print("\n=== 3. ENROLLMENT vs COMPLETION ===")
d = df[df["enrollment"].notna() & (df["enrollment"] >= 0)].copy()
d["bucket"] = pd.qcut(d["enrollment"].clip(upper=d["enrollment"].quantile(0.99)),
                      q=5, duplicates="drop")
print(d.groupby("bucket", observed=True)["completed"].mean().map(pct).to_string())

print("\n=== 4. PHASE x SPONSOR (lowest & highest cells) ===")
piv = df.pivot_table(index="phase", columns="sponsor_class",
                     values="completed", aggfunc="mean")
stacked = piv.stack().sort_values()
print("Lowest cell:", stacked.index[0], pct(stacked.iloc[0]))
print("Highest cell:", stacked.index[-1], pct(stacked.iloc[-1]))

print("\n=== 5. TREND OVER TIME ===")
if "start_year" in df:
    yr = pd.to_numeric(df["start_year"], errors="coerce")
    t = df.assign(y=yr).dropna(subset=["y"])
    t = t[t["y"].between(2008, 2020)].groupby("y")["completed"].mean()
    if len(t) > 1:
        direction = "up" if t.iloc[-1] > t.iloc[0] else "down"
        print(f"{int(t.index[0])}: {pct(t.iloc[0])}  ->  {int(t.index[-1])}: {pct(t.iloc[-1])}  "
              f"(trend: {direction})")

print("\nCopy these into the ___ blanks in dashboard/app.py, then rerun the app.\n")
