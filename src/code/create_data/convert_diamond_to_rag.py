"""
Converts Diamond tier champion data into RAG chunks.
Uses only the latest patch (16_9) to keep the index focused.

Creates: diamond_builds.jsonl
- Per-champion, per-role build data (latest patch only)
- Summary leaderboard chunks (top WR, most played, popular runes per role)

Usage:
    python convert_diamond_to_rag.py
"""

import json
import os
from typing import List, Dict
from collections import Counter, defaultdict

_HERE = os.path.dirname(os.path.abspath(__file__))
_DATA = os.path.join(_HERE, "..", "..", "data")

LATEST_PATCH = "16_9"
MIN_MATCHES = 500  # filter out troll/off-meta picks with tiny sample sizes


def load_patch_data(patch: str) -> Dict:
    """Load diamond data for a specific patch."""
    filepath = os.path.join(_DATA, f"diamond_{patch}_named.json")
    if not os.path.exists(filepath):
        print(f"Warning: {filepath} not found, skipping...")
        return {}
    
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def create_build_chunk(champ_name: str, role: str, patch: str, role_data: Dict) -> Dict:
    """Create a RAG chunk for champion build data."""
    overview = role_data.get("overview", {})
    summary = overview.get("summary", {})
    
    # Extract runes
    perks = overview.get("perks", {})
    runes_named = perks.get("runes_named", [])
    rune_names = [r["name"] for r in runes_named] if runes_named else []
    primary_tree = perks.get("primary_tree_name", "")
    secondary_tree = perks.get("secondary_tree_name", "")
    
    # Extract summoner spells
    spells_data = overview.get("summoner_spells", {})
    spells_named = spells_data.get("spells_named", [])
    spell_names = [s["name"] for s in spells_named] if spells_named else []
    
    # Extract items
    starting = overview.get("starting_items", {})
    starting_items = [i["name"] for i in starting.get("items_named", [])]
    
    core = overview.get("core_items", {})
    core_items = [i["name"] for i in core.get("items_named", [])]
    
    # Skill path
    skill_path = overview.get("skill_path", {})
    skill_order = skill_path.get("priority", "")
    full_order = " -> ".join(skill_path.get("order", []))
    
    # Win rate and sample size
    win_rate = summary.get("win_rate", 0)
    matches = summary.get("matches", 0)
    
    # Build item options text
    item_options = role_data.get("item_options", [])
    item_options_text = []
    for slot_data in item_options:
        slot = slot_data.get("slot", "")
        options = slot_data.get("options", [])
        if options:
            option_texts = []
            for opt in options[:3]:  # Top 3 options
                opt_name = opt.get("item_name", "")
                opt_wr = opt.get("win_rate", 0)
                opt_matches = opt.get("matches", 0)
                option_texts.append(f"{opt_name} ({opt_wr:.1%} WR, {opt_matches} games)")
            item_options_text.append(f"{slot}: {', '.join(option_texts)}")
    
    # Create searchable text
    text_parts = [
        f"Champion: {champ_name}",
        f"Role: {role}",
        f"Patch: {patch}",
        f"Rank: Diamond",
        "",
        f"Win Rate: {win_rate:.1%} ({matches:,} matches)",
        "",
        f"Runes: {primary_tree} + {secondary_tree}",
        f"Primary Runes: {', '.join(rune_names[:4])}",
        f"Secondary Runes: {', '.join(rune_names[4:])}",
        "",
        f"Summoner Spells: {', '.join(spell_names)}",
        "",
        f"Starting Items: {', '.join(starting_items)}",
        f"Core Build: {', '.join(core_items)}",
        "",
        f"Skill Priority: {skill_order}",
        f"Skill Order: {full_order}",
    ]
    
    if item_options_text:
        text_parts.extend(["", "Situational Items:"] + item_options_text)
    
    text = "\n".join(text_parts)
    
    # Metadata for filtering
    metadata = {
        "champion": champ_name,
        "role": role,
        "patch": patch,
        "rank": "diamond",
        "win_rate": win_rate,
        "matches": matches,
        "runes": rune_names,
        "primary_tree": primary_tree,
        "secondary_tree": secondary_tree,
        "spells": spell_names,
        "core_items": core_items,
        "skill_priority": skill_order,
    }
    
    return {
        "text": text,
        "metadata": metadata,
    }


def process_latest_patch() -> List[Dict]:
    """Process only the latest patch and create chunks."""
    print(f"\nProcessing patch {LATEST_PATCH} (latest only)...")
    data = load_patch_data(LATEST_PATCH)
    
    if not data:
        return []
    
    chunks = []
    skipped = 0
    champions = data.get("champions", {})
    
    for champ_slug, champ_data in champions.items():
        champ_name = champ_data.get("name", champ_slug)
        roles = champ_data.get("roles", {})
        
        for role, role_data in roles.items():
            if not role_data.get("overview"):
                continue
            overview = role_data["overview"]
            matches = overview.get("summary", {}).get("matches", 0)
            if matches < MIN_MATCHES:
                skipped += 1
                continue
            
            chunk = create_build_chunk(champ_name, role, LATEST_PATCH, role_data)
            chunks.append(chunk)
    
    print(f"  Created {len(chunks)} build chunks (skipped {skipped} with <{MIN_MATCHES} matches)")
    return chunks


def create_summary_chunks(build_chunks: List[Dict]) -> List[Dict]:
    """Create aggregated leaderboard chunks for ranking questions."""
    summaries = []
    
    # Group chunks by role
    by_role = defaultdict(list)
    for c in build_chunks:
        by_role[c["metadata"]["role"]].append(c["metadata"])
    
    for role, champs in by_role.items():
        role_title = role.upper() if role == "adc" else role.capitalize()
        
        # ── Top 10 highest win rate per role ──
        top_wr = sorted(champs, key=lambda x: x["win_rate"], reverse=True)[:10]
        lines = [f"Top 10 highest win rate {role_title} champions in Diamond (Patch {LATEST_PATCH}):"]
        for i, c in enumerate(top_wr, 1):
            lines.append(f"{i}. {c['champion']} {role_title} - {c['win_rate']:.1%} WR ({c['matches']:,} matches)")
        summaries.append({
            "text": "\n".join(lines),
            "metadata": {"champion": f"Top WR {role_title}", "role": role, "patch": LATEST_PATCH,
                         "rank": "diamond", "win_rate": 0, "matches": 0,
                         "runes": [], "primary_tree": "", "secondary_tree": "",
                         "spells": [], "core_items": [], "skill_priority": ""},
        })
        
        # ── Top 10 lowest win rate per role ──
        bot_wr = sorted(champs, key=lambda x: x["win_rate"])[:10]
        lines = [f"Top 10 lowest win rate {role_title} champions in Diamond (Patch {LATEST_PATCH}):"]
        for i, c in enumerate(bot_wr, 1):
            lines.append(f"{i}. {c['champion']} {role_title} - {c['win_rate']:.1%} WR ({c['matches']:,} matches)")
        summaries.append({
            "text": "\n".join(lines),
            "metadata": {"champion": f"Lowest WR {role_title}", "role": role, "patch": LATEST_PATCH,
                         "rank": "diamond", "win_rate": 0, "matches": 0,
                         "runes": [], "primary_tree": "", "secondary_tree": "",
                         "spells": [], "core_items": [], "skill_priority": ""},
        })
        
        # ── Top 10 most played per role ──
        top_played = sorted(champs, key=lambda x: x["matches"], reverse=True)[:10]
        lines = [f"Top 10 most played {role_title} champions in Diamond (Patch {LATEST_PATCH}):"]
        for i, c in enumerate(top_played, 1):
            lines.append(f"{i}. {c['champion']} {role_title} - {c['matches']:,} matches ({c['win_rate']:.1%} WR)")
        summaries.append({
            "text": "\n".join(lines),
            "metadata": {"champion": f"Most Played {role_title}", "role": role, "patch": LATEST_PATCH,
                         "rank": "diamond", "win_rate": 0, "matches": 0,
                         "runes": [], "primary_tree": "", "secondary_tree": "",
                         "spells": [], "core_items": [], "skill_priority": ""},
        })
        
        # ── Most popular rune keystones per role ──
        keystone_counter = Counter()
        for c in champs:
            if c["runes"]:
                keystone_counter[c["runes"][0]] += c["matches"]
        top_keystones = keystone_counter.most_common(5)
        lines = [f"Most popular rune keystones for {role_title} in Diamond (Patch {LATEST_PATCH}):"]
        total = sum(v for _, v in top_keystones)
        for i, (ks, count) in enumerate(top_keystones, 1):
            lines.append(f"{i}. {ks} - {count:,} matches ({count/total:.0%} of {role_title} games)")
        summaries.append({
            "text": "\n".join(lines),
            "metadata": {"champion": f"Keystones {role_title}", "role": role, "patch": LATEST_PATCH,
                         "rank": "diamond", "win_rate": 0, "matches": 0,
                         "runes": [], "primary_tree": "", "secondary_tree": "",
                         "spells": [], "core_items": [], "skill_priority": ""},
        })
        
        # ── Most common starting items per role ──
        item_counter = Counter()
        for c in champs:
            for item in c["core_items"]:
                item_counter[item] += c["matches"]
        top_items = item_counter.most_common(10)
        lines = [f"Most popular core items for {role_title} in Diamond (Patch {LATEST_PATCH}):"]
        for i, (item, count) in enumerate(top_items, 1):
            lines.append(f"{i}. {item} - built in {count:,} matches")
        summaries.append({
            "text": "\n".join(lines),
            "metadata": {"champion": f"Core Items {role_title}", "role": role, "patch": LATEST_PATCH,
                         "rank": "diamond", "win_rate": 0, "matches": 0,
                         "runes": [], "primary_tree": "", "secondary_tree": "",
                         "spells": [], "core_items": [], "skill_priority": ""},
        })
    
    print(f"  Created {len(summaries)} summary/leaderboard chunks")
    return summaries


def main():
    print("Converting Diamond tier patch data to RAG chunks...")
    print(f"Using latest patch only: {LATEST_PATCH}")
    print(f"Minimum matches filter: {MIN_MATCHES}")
    
    build_chunks = process_latest_patch()
    
    if not build_chunks:
        print("ERROR: No build chunks created!")
        return
    
    summary_chunks = create_summary_chunks(build_chunks)
    all_chunks = build_chunks + summary_chunks
    
    # Save to JSONL
    output_file = os.path.join(_DATA, "diamond_builds.jsonl")
    with open(output_file, "w", encoding="utf-8") as f:
        for chunk in all_chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")
    
    print(f"\n✓ Saved {len(all_chunks)} total chunks to {output_file}")
    print(f"    Build chunks:   {len(build_chunks)}")
    print(f"    Summary chunks: {len(summary_chunks)}")
    
    # Show samples
    if build_chunks:
        print("\n  Sample build chunk:")
        sample = build_chunks[0]
        meta = sample["metadata"]
        print(f"    Champion: {meta['champion']}, Role: {meta['role']}, Patch: {meta['patch']}")
        print("    " + sample["text"].replace("\n", "\n    "))
    
    if summary_chunks:
        print("\n  Sample summary chunk:")
        print("    " + summary_chunks[0]["text"].replace("\n", "\n    "))
    
    # Statistics
    champions = set(c["metadata"]["champion"] for c in build_chunks)
    roles = set(c["metadata"]["role"] for c in build_chunks)
    
    print(f"\n  Statistics:")
    print(f"    Patch: {LATEST_PATCH}")
    print(f"    Champions: {len(champions)}")
    print(f"    Roles: {len(roles)} ({', '.join(sorted(roles))})")
    print(f"    Min matches filter: {MIN_MATCHES}")


if __name__ == "__main__":
    main()
