"""Generate the Phase 7 local NFT metadata files for the seeded MREID subset.

Reads scripts/millowSeed.json (easy to extend: add an MREID id and rerun),
looks up the non-sensitive listing facts in data/processed/MREID_property.csv,
and writes public/metadata/millow/<mreid_id>.json.

Only non-sensitive, already-public property attributes are written. No pricing
economics, no owner/legal/personal data, nothing sensitive.
"""

import csv
import json
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except AttributeError:
    pass

ROOT = Path(__file__).resolve().parents[1]
SEED_PATH = ROOT / "scripts" / "millowSeed.json"
CSV_PATH = ROOT / "data" / "processed" / "MREID_property.csv"
OUT_DIR = ROOT / "public" / "metadata" / "millow"

DESCRIPTION = (
    "Digital property representation of a MILLOW MREID listing "
    "(technical proof-of-concept). Blockchain-based ownership record for the "
    "prototype. Legal ownership remains subject to applicable Indian property "
    "and registration law."
)


def load_rows():
    rows = {}
    with open(CSV_PATH, encoding="utf-8", errors="replace") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows[row["mreid_id"]] = row
    return rows


def main():
    seed = json.loads(SEED_PATH.read_text(encoding="utf-8"))
    rows = load_rows()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for mreid in seed:
        row = rows.get(mreid)
        if row is None:
            print(f"WARN: {mreid} not found in MREID dataset, skipped")
            continue

        metadata = {
            "name": f"MILLOW Property {mreid}",
            "property_identifier": mreid,
            "description": DESCRIPTION,
            "attributes": [
                {"trait_type": "City", "value": row["source_city"]},
                {"trait_type": "Location", "value": row.get("location", "")},
                {"trait_type": "Bedrooms", "value": int(row.get("no_of_bedrooms", 0))},
                {"trait_type": "Area (sqft)", "value": float(row.get("area", 0))},
            ],
        }
        target = OUT_DIR / f"{mreid}.json"
        target.write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(f"Wrote {target.relative_to(ROOT)}")

    print(f"Done. {len(seed)} seed ids processed.")


if __name__ == "__main__":
    main()