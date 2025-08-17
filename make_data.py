# make_data.py
from pathlib import Path
import argparse
import random
import csv

ROOT = Path(__file__).resolve().parent
SEED_DIR = ROOT / "data" / "seed"
SEED_DIR.mkdir(parents=True, exist_ok=True)

# ----------------- Vocab -----------------
fund_firms = [
    "Arcadia",
    "BluePeak",
    "NorthBridge",
    "Sierra",
    "Atlas",
    "Crescent",
    "RiverRock",
    "Summit",
    "Granite",
    "Evergreen",
    "Pioneer",
    "Liberty",
]
adjectives = [
    "Global",
    "International",
    "Strategic",
    "Core",
    "Dynamic",
    "Enhanced",
    "Balanced",
    "Growth",
    "Value",
    "Prime",
    "Selective",
]
assets = [
    "Equity",
    "Bond",
    "Income",
    "Dividend",
    "Index",
    "Multi-Asset",
    "Short Term",
    "Small Cap",
    "Mid Cap",
    "Large Cap",
    "Treasury",
]
wrappers = ["Fund", "Trust", "UCITS", "ETF", "Institutional Fund"]
regions = ["US", "Europe", "Asia Pacific", "EM", "World", "Global"]
strategies = [
    "Quality",
    "Momentum",
    "Minimum Volatility",
    "High Yield",
    "Aggregate Bond",
    "Total Market",
    "Value",
    "Growth",
]

companies = [
    "Acme",
    "Globex",
    "Initech",
    "Umbrella",
    "Hooli",
    "Stark",
    "Wayne",
    "Wonka",
    "Soylent",
    "BlackRock",
    "Tesla",
    "Apple",
    "Contoso",
    "Aperture",
    "Cyberdyne",
    "Vandelay",
    "Massive Dynamic",
    "Tyrell",
    "Nakatomi",
]
company_suffix = ["Inc.", "LLC", "Ltd.", "Corporation", "Group", "Holdings", "PLC"]
departments = ["Finance", "HR", "IT", "Sales", "Marketing", "Support", "Operations"]
locations = ["New York", "London", "Tokyo", "Singapore", "Sydney", "Berlin", "Toronto"]
people_first = [
    "John",
    "Sarah",
    "Alex",
    "Minh",
    "Olivia",
    "Emma",
    "Liam",
    "Ava",
    "Noah",
    "Mason",
    "Ethan",
    "Mia",
    "Lucas",
    "Sofia",
    "Daniel",
    "Chloe",
    "Isabella",
    "James",
]
people_last = [
    "Nguyen",
    "Smith",
    "Johnson",
    "Brown",
    "Lee",
    "Davis",
    "Wilson",
    "Taylor",
    "Clark",
    "Garcia",
    "Anderson",
    "Martinez",
    "Hall",
    "Lopez",
]
generic_terms = [
    "Invoice",
    "Wire Transfer Receipt",
    "Account Statement",
    "Purchase Order",
    "Onboarding Guide",
    "IT Support Ticket",
    "Travel Reimbursement",
    "Product Roadmap",
    "Quarterly Earnings Call",
    "Expense Report",
    "Meeting Notes",
    "Job Application",
]
banks = [
    "Bank of America",
    "Chase",
    "Wells Fargo",
    "Citibank",
    "Capital One",
    "PNC",
    "HSBC",
]

FUND_TEMPLATES = [
    "{firm} {adj} {asset} {wrap}",
    "{region} {asset} {wrap}",
    "{firm} {strategy} {wrap}",
    "{firm} {asset} Index {wrap}",
    "{firm} {adj} {wrap}",
]

# More diverse Non-Fund templates
NONFUND_TEMPLATES = [
    "{co} {suf}",
    "{co} {suf} – Careers",
    "{co} {suf} – Headquarters",
    "{co} {suf} {dept}",
    "{co} {suf} {dept} {loc}",
    "{first} {last}",
    "{first} {last} – {dept}",
    "{bank} checking account",
    "{generic}",
    "{generic} #{num4}",
    "{co} {suf} Ref {num6}",
    "Ticket {num6} – {dept}",
    "{dept} Request {num5}",
    "Email thread: {generic} ({num4})",
    "{co} {suf} Vendor ID {num5}",
]


# ----------------- Helpers -----------------
def normalize(s: str) -> str:
    return " ".join(str(s).lower().strip().split())


def synth_fund(rng: random.Random) -> str:
    tmpl = rng.choice(FUND_TEMPLATES)
    name = tmpl.format(
        firm=rng.choice(fund_firms),
        adj=rng.choice(adjectives),
        asset=rng.choice(assets),
        wrap=rng.choice(wrappers),
        region=rng.choice(regions),
        strategy=rng.choice(strategies),
    )
    if rng.random() < 0.12:
        name = name.replace("  ", " ").strip()
    return name


def _num(rng: random.Random, k: int) -> str:
    lo = 10 ** (k - 1)
    hi = 10**k - 1
    return str(rng.randint(lo, hi))


def synth_nonfund(rng: random.Random) -> str:
    tmpl = rng.choice(NONFUND_TEMPLATES)
    return tmpl.format(
        co=rng.choice(companies),
        suf=rng.choice(company_suffix),
        dept=rng.choice(departments),
        loc=rng.choice(locations),
        first=rng.choice(people_first),
        last=rng.choice(people_last),
        bank=rng.choice(banks),
        generic=rng.choice(generic_terms),
        num4=_num(rng, 4),
        num5=_num(rng, 5),
        num6=_num(rng, 6),
    )


def generate_unique(label: str, n: int, rng: random.Random, max_tries: int = 100000):
    """
    Try natural generation first; if we fall short, deterministically add a numeric 'salt'
    to ensure uniqueness while keeping strings realistic.
    """
    rows, seen, tries = [], set(), 0
    maker = synth_fund if label == "Fund" else synth_nonfund

    # Phase 1: natural generation
    while len(rows) < n and tries < max_tries:
        tries += 1
        name = maker(rng).strip()
        key = normalize(name)
        if key not in seen:
            rows.append((label, name))
            seen.add(key)

    # Phase 2: guaranteed uniqueness (only if needed for Non-Fund or Fund)
    while len(rows) < n:
        base = maker(rng).strip()
        # add a small numeric token that should survive normalization
        salt = _num(rng, 6)
        name = f"{base} #{salt}"
        key = normalize(name)
        if key not in seen:
            rows.append((label, name))
            seen.add(key)

    return rows, tries


# ----------------- Main -----------------
def main(n_per_class: int, seed: int):
    rng = random.Random(seed)
    rng.shuffle(FUND_TEMPLATES)
    rng.shuffle(NONFUND_TEMPLATES)

    fund_rows, fund_tries = generate_unique("Fund", n_per_class, rng)
    nonfund_rows, nonfund_tries = generate_unique("Non-Fund", n_per_class, rng)

    total = len(fund_rows) + len(nonfund_rows)

    out_path = SEED_DIR / "fund_nonfund.csv"
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["label", "name"])
        # interleave for variety
        for i in range(n_per_class):
            w.writerow(fund_rows[i])
            w.writerow(nonfund_rows[i])

    print(f"Wrote {total} rows → {out_path}")
    print(f"Phase-1 attempts: Fund={fund_tries}, Non-Fund={nonfund_tries}")
    print("Sample Fund:", fund_rows[:3])
    print("Sample Non-Fund:", nonfund_rows[:3])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--n-per-class", type=int, default=600, help="Unique rows per class"
    )
    ap.add_argument("--seed", type=int, default=42, help="Random seed")
    args = ap.parse_args()
    main(n_per_class=args.n_per_class, seed=args.seed)
