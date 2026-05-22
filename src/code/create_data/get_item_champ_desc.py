"""
Fetches champion and item descriptions from Riot Data Dragon (no API key needed).
Outputs two JSONL files ready to be added to the RAG FAISS index:
  - src/data/champion_chunks.jsonl
  - src/data/item_chunks.jsonl

Usage:
    python get_item_champ_desc.py
    python get_item_champ_desc.py --version 16.10.1
"""

import argparse
import json
import os
import re
import time
import urllib.request

_HERE   = os.path.dirname(os.path.abspath(__file__))
_DATA   = os.path.join(_HERE, "..", "..", "data")

parser = argparse.ArgumentParser()
parser.add_argument("--version", default=None, help="Data Dragon patch version e.g. 16.10.1 (default: latest)")
args = parser.parse_args()

def fetch(url):
    print(f"  GET {url}")
    with urllib.request.urlopen(url, timeout=30) as r:
        return json.loads(r.read().decode())

# ── Get latest version ────────────────────────────────────────────────────────
if args.version:
    VERSION = args.version
else:
    versions = fetch("https://ddragon.leagueoflegends.com/api/versions.json")
    VERSION  = versions[0]
print(f"Using Data Dragon version: {VERSION}\n")

BASE = f"https://ddragon.leagueoflegends.com/cdn/{VERSION}/data/en_US"

# ── Helper: strip HTML tags from descriptions ─────────────────────────────────
def strip_html(text):
    return re.sub(r"<[^>]+>", "", text or "").strip()

# ══════════════════════════════════════════════════════════════════════════════
# CHAMPIONS
# ══════════════════════════════════════════════════════════════════════════════
print("=== Fetching champion list ===")
champ_summary = fetch(f"{BASE}/champion.json")
champ_keys    = list(champ_summary["data"].keys())
print(f"Found {len(champ_keys)} champions")

champ_chunks = []

for champ_id in champ_keys:
    try:
        data = fetch(f"{BASE}/champion/{champ_id}.json")
        c    = data["data"][champ_id]
    except Exception as e:
        print(f"  [warn] failed to fetch {champ_id}: {e}")
        continue

    name   = c["name"]
    title  = c["title"]
    tags   = ", ".join(c.get("tags", []))
    blurb  = c.get("blurb", "")
    tips_ally  = "; ".join(c.get("allytips", []))
    tips_enemy = "; ".join(c.get("enemytips", []))
    passive    = c.get("passive", {})
    spells     = c.get("spells", [])

    abilities_text = f"Passive — {passive.get('name','')}: {strip_html(passive.get('description',''))}\n"
    for spell in spells:
        abilities_text += f"{spell['name']}: {strip_html(spell.get('description',''))}\n"

    text = (
        f"Champion: {name} — {title}\n"
        f"Roles: {tags}\n"
        f"Description: {blurb}\n"
        f"Abilities:\n{abilities_text.strip()}\n"
        f"Tips when playing {name}: {tips_ally}\n"
        f"Tips when playing against {name}: {tips_enemy}"
    ).strip()

    champ_chunks.append({
        "id":      f"champion__{champ_id.lower()}",
        "subject": name,
        "type":    "champion",
        "roles":   c.get("tags", []),
        "text":    text,
    })
    time.sleep(0.05)

out_champ = os.path.join(_DATA, "champion_chunks.jsonl")
with open(out_champ, "w", encoding="utf-8") as f:
    for chunk in champ_chunks:
        f.write(json.dumps(chunk, ensure_ascii=False) + "\n")
print(f"\nSaved {len(champ_chunks)} champion chunks → {out_champ}")

# ══════════════════════════════════════════════════════════════════════════════
# ITEMS
# ══════════════════════════════════════════════════════════════════════════════
print("\n=== Fetching items ===")
item_data = fetch(f"{BASE}/item.json")
items_raw = item_data["data"]
print(f"Found {len(items_raw)} items")

item_chunks = []

for item_id, item in items_raw.items():
    name = item.get("name", "")
    if not name or name.startswith("Enchantment"):
        continue

    desc    = strip_html(item.get("description", ""))
    tags    = ", ".join(item.get("tags", []))
    gold    = item.get("gold", {})
    cost    = gold.get("total", 0)
    stats   = item.get("stats", {})

    stats_text = ""
    if stats:
        stats_text = "Stats: " + ", ".join(f"{k}: {v}" for k, v in stats.items())

    text = (
        f"Item: {name}\n"
        f"Tags: {tags}\n"
        f"Gold cost: {cost}\n"
        f"Description: {desc}\n"
        f"{stats_text}"
    ).strip()

    item_chunks.append({
        "id":      f"item__{item_id}",
        "subject": name,
        "type":    "item",
        "tags":    item.get("tags", []),
        "cost":    cost,
        "text":    text,
    })

out_item = os.path.join(_DATA, "item_chunks.jsonl")
with open(out_item, "w", encoding="utf-8") as f:
    for chunk in item_chunks:
        f.write(json.dumps(chunk, ensure_ascii=False) + "\n")
print(f"Saved {len(item_chunks)} item chunks → {out_item}")

print("\nDone. Next step: re-index the FAISS store with these chunks added.")
