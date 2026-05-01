import os
import sys
import csv
import json
import time
import requests

API_KEY = os.getenv("RIOT_API_KEY")
if not API_KEY:
    raise ValueError("Set RIOT_API_KEY environment variable.")

HEADERS = {"X-Riot-Token": API_KEY}
MATCHES_PER_PLAYER = 25
QUEUE_ID = 420  # Ranked Solo/Duo only

def riot_get(url, params=None, max_retries=5):
    """Generic Riot API GET with rate limit handling."""
    backoff = 1
    for attempt in range(max_retries):
        try:
            r = requests.get(url, headers=HEADERS, params=params, timeout=10)
            if r.status_code == 200:
                return r.json()
            elif r.status_code == 429:
                wait = int(r.headers.get("Retry-After", backoff))
                print(f"  [429] Rate limit. Waiting {wait}s...")
                time.sleep(wait)
                backoff *= 2
            elif r.status_code in (500, 503):
                print(f"  [{r.status_code}] Server error. Waiting {backoff}s...")
                time.sleep(backoff)
                backoff *= 2
            else:
                print(f"  [{r.status_code}] {url}")
                return None
        except requests.exceptions.RequestException as e:
            print(f"  Network error: {e}. Waiting {backoff}s...")
            time.sleep(backoff)
            backoff *= 2
    return None

def get_match_ids(puuid):
    url = f"https://europe.api.riotgames.com/lol/match/v5/matches/by-puuid/{puuid}/ids"
    return riot_get(url, params={"count": MATCHES_PER_PLAYER, "queue": QUEUE_ID}) or []

def get_match_details(match_id):
    url = f"https://europe.api.riotgames.com/lol/match/v5/matches/{match_id}"
    return riot_get(url)

def extract_participant(match_data, puuid):
    """Extract only the fields we need for finetuning."""
    info = match_data["info"]
    participants = info["participants"]

    for p in participants:
        if p["puuid"] != puuid:
            continue

        # Items (slots 0-5, exclude 0=empty and 6=trinket)
        items = [p.get(f"item{i}", 0) for i in range(6)]
        items = [i for i in items if i != 0]

        # Enemy champions
        team_id = p["teamId"]
        enemies = [x["championName"] for x in participants if x["teamId"] != team_id]

        # Ally champions (excluding self)
        allies = [x["championName"] for x in participants
                  if x["teamId"] == team_id and x["puuid"] != puuid]

        # Clean patch version (e.g. "14.10.123.456" → "14.10")
        raw_patch = info.get("gameVersion", "")
        patch = ".".join(raw_patch.split(".")[:2]) if raw_patch else ""

        return {
            "patch":            patch,
            "game_duration_min": round(info["gameDuration"] / 60, 1),
            "champion":         p["championName"],
            "position":         p["teamPosition"],   # TOP, JG, MID, BOT, SUPPORT
            "win":              p["win"],
            "kills":            p["kills"],
            "deaths":           p["deaths"],
            "assists":          p["assists"],
            "cs":               p["totalMinionsKilled"] + p["neutralMinionsKilled"],
            "gold":             p["goldEarned"],
            "items":            items,               # list of item IDs
            "enemy_team":       enemies,             # list of champion names
            "ally_team":        allies,
            "damage_dealt":     p["totalDamageDealtToChampions"],
            "vision_score":     p["visionScore"],
            "first_blood":      p["firstBloodKill"],
        }
    return None

def main(input_csv, output_json):
    all_matches = []
    seen_match_ids = set()

    # Load already-saved matches if resuming
    if os.path.exists(output_json):
        with open(output_json, "r", encoding="utf-8") as f:
            all_matches = json.load(f)
        print(f"Resuming — loaded {len(all_matches)} existing records.")

    with open(input_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        players = list(reader)

    print(f"Processing {len(players)} players...")

    for i, row in enumerate(players):
        puuid = row.get("puuid", "").strip()
        name  = row.get("Name", "")
        if not puuid:
            continue

        print(f"[{i+1}/{len(players)}] {name} — fetching match IDs...")
        match_ids = get_match_ids(puuid)
        time.sleep(1.2)

        new_count = 0
        for match_id in match_ids:
            if match_id in seen_match_ids:
                continue
            seen_match_ids.add(match_id)

            match_data = get_match_details(match_id)
            time.sleep(1.2)
            if not match_data:
                continue

            entry = extract_participant(match_data, puuid)
            if entry:
                all_matches.append(entry)
                new_count += 1

        print(f"  +{new_count} new matches | Total: {len(all_matches)}")

        # Save progress every 50 players (crash-safe)
        if (i + 1) % 50 == 0:
            with open(output_json, "w", encoding="utf-8") as f:
                json.dump(all_matches, f, indent=2)
            print(f"  [checkpoint] Saved {len(all_matches)} records.")

    # Final save
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(all_matches, f, indent=2)

    print(f"\nDone. {len(all_matches)} match records saved to {output_json}.")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python getMatchData.py <puuids.csv> <output.json>")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])