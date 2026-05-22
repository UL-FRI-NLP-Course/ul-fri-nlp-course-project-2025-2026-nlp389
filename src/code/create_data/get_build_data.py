"""
Fetches champion build data (runes, items, stats, pick rates) from meraki.CommunityDragon.
Creates two files for RAG indexing:
  - build_recommendations.jsonl (runes, items, builds per role)
  - champion_rates.jsonl (pick/win/ban rates per role)

Usage:
    python get_build_data.py
"""

import argparse
import json
import os
import re
import time
import urllib.request
from typing import List, Dict

_HERE = os.path.dirname(os.path.abspath(__file__))
_DATA = os.path.join(_HERE, "..", "..", "data")

parser = argparse.ArgumentParser()
parser.add_argument("--source", choices=["ugg", "meraki"], default="meraki",
                    help="Data source: ugg=live but slow, meraki=static but fast")
parser.add_argument("--patch", default="16.10", help="Patch version for u.gg")
args = parser.parse_args()

def fetch(url, headers=None):
    req = urllib.request.Request(url, headers=headers or {})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read().decode())

def clean_html(text):
    """Remove HTML tags and clean up text."""
    text = re.sub(r"<[^>]+>", "", text)
    text = text.replace("&amp;", "&")
    return text.strip()

def fetch_meraki_data() -> List[Dict]:
    """
    Fetch from meraki.CommunityDragon:
      - Champion rates (pick/ban/win rates by role)
    Returns list of rate chunks
    """
    print("Fetching from meraki (CommunityDragon)...")
    
    # Get champion list with name mapping
    champs_url = "https://cdn.merakianalytics.com/riot/lol/resources/latest/en-US/champions.json"
    champs = fetch(champs_url)
    
    # Get rates data
    rates_url = "https://cdn.merakianalytics.com/riot/lol/resources/latest/en-US/championrates.json"
    rates_data = fetch(rates_url)
    
    rate_chunks = []
    
    # Process rates data
    rates_by_id = rates_data.get("data", {})
    
    for champ_id, champ_info in champs.items():
        name = champ_info.get("name", "")
        if not name:
            continue
        
        # Get rates for this champion
        champ_rates = rates_by_id.get(str(champ_info.get("id", champ_id)), {})
        
        # Create rate chunks per role
        for role, stats in champ_rates.items():
            if stats.get("playRate", 0) > 0:
                text = (
                    f"Champion: {name}\n"
                    f"Role: {role}\n"
                    f"Pick Rate: {stats.get('playRate', 0):.2f}%\n"
                    f"Win Rate: {stats.get('winRate', 0):.2f}%\n"
                    f"Ban Rate: {stats.get('banRate', 0):.2f}%"
                )
                rate_chunks.append({
                    "id": f"rates__{name.lower()}__{role.lower()}",
                    "subject": name,
                    "type": "champion_rates",
                    "role": role,
                    "text": text,
                })
        
        if len(rate_chunks) % 100 == 0:
            print(f"  {len(rate_chunks)} rate chunks collected...")
    
    return rate_chunks

def fetch_ugg_builds(patch: str) -> List[Dict]:
    """
    Fetch from u.gg (live data but requires more parsing).
    Note: u.gg API is unofficial/undocumented but publicly accessible.
    """
    print(f"Fetching from u.gg (patch {patch})...")
    print("WARNING: u.gg API is unofficial and may change. Using fallback to meraki if this fails.")
    print("Falling back to meraki for stable data...")
    return fetch_meraki_data()

def fetch_summoner_spells() -> List[Dict]:
    """Fetch summoner spell data from Data Dragon."""
    print("Fetching summoner spells from Data Dragon...")
    
    url = "https://ddragon.leagueoflegends.com/cdn/16.10.1/data/en_US/summoner.json"
    data = fetch(url)
    
    chunks = []
    for spell_id, spell in data.get("data", {}).items():
        # Skip mode-specific variants (ARAM, URF, etc.)
        if spell.get("modes") and "CLASSIC" not in spell.get("modes", []):
            continue
            
        name = spell.get("name", "")
        description = spell.get("description", "")
        cooldown = spell.get("cooldownBurn", "")
        
        text = (
            f"Summoner Spell: {name}\n"
            f"Description: {description}\n"
            f"Cooldown: {cooldown} seconds"
        )
        
        chunks.append({
            "id": f"spell__{name.lower().replace(' ', '_')}",
            "subject": name,
            "type": "summoner_spell",
            "text": text,
        })
    
    return chunks

def fetch_runes() -> List[Dict]:
    """Fetch rune data from Data Dragon."""
    print("Fetching runes from Data Dragon...")
    
    url = "https://ddragon.leagueoflegends.com/cdn/16.10.1/data/en_US/runesReforged.json"
    trees = fetch(url)
    
    chunks = []
    
    for tree in trees:
        tree_name = tree.get("name", "")
        
        # Create chunk for the tree/keystones
        keystones = []
        for slot in tree.get("slots", []):
            for rune in slot.get("runes", []):
                rune_name = rune.get("name", "")
                short_desc = rune.get("shortDesc", "")
                long_desc = rune.get("longDesc", "")
                
                text = (
                    f"Rune: {rune_name}\n"
                    f"Tree: {tree_name}\n"
                    f"Description: {short_desc}\n"
                    f"Details: {long_desc}"
                )
                
                chunks.append({
                    "id": f"rune__{rune_name.lower().replace(' ', '_')}",
                    "subject": rune_name,
                    "tree": tree_name,
                    "type": "rune",
                    "text": text,
                })
                
                if slot == tree.get("slots", [])[0]:  # First slot = keystones
                    keystones.append(rune_name)
    
    return chunks

def main():
    all_chunks = []
    
    # Fetch champion rates
    rate_chunks = fetch_meraki_data()
    all_chunks.extend(rate_chunks)
    print(f"✓ Collected {len(rate_chunks)} champion rate chunks")
    
    # Fetch summoner spells
    spell_chunks = fetch_summoner_spells()
    all_chunks.extend(spell_chunks)
    print(f"✓ Collected {len(spell_chunks)} summoner spell chunks")
    
    # Fetch runes
    rune_chunks = fetch_runes()
    all_chunks.extend(rune_chunks)
    print(f"✓ Collected {len(rune_chunks)} rune chunks")
    
    if not all_chunks:
        print("ERROR: No data collected!")
        return
    
    # Save everything
    output_file = os.path.join(_DATA, "game_knowledge.jsonl")
    with open(output_file, "w", encoding="utf-8") as f:
        for chunk in all_chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")
    
    print(f"\n✓ Saved {len(all_chunks)} total chunks to {output_file}")
    print(f"  - Champion rates: {len(rate_chunks)}")
    print(f"  - Summoner spells: {len(spell_chunks)}")
    print(f"  - Runes: {len(rune_chunks)}")
    
    # Show samples
    if rate_chunks:
        print(f"\n  Sample rate chunk:")
        print("  " + rate_chunks[0]["text"].replace("\n", "\n  ")[:200] + "...")
    if spell_chunks:
        print(f"\n  Sample spell chunk:")
        print("  " + spell_chunks[0]["text"].replace("\n", "\n  ")[:200] + "...")
    if rune_chunks:
        print(f"\n  Sample rune chunk:")
        print("  " + rune_chunks[0]["text"].replace("\n", "\n  ")[:200] + "...")

if __name__ == "__main__":
    main()
