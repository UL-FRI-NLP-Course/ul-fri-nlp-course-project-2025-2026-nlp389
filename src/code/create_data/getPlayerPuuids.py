import os
import sys
import csv
import time
import requests

API_KEY = os.getenv("RIOT_API_KEY")
if not API_KEY:
    raise ValueError("API key not found. Please set the RIOT_API_KEY environment variable.")

HEADERS = {"X-Riot-Token": API_KEY}
BASE_URL = "https://europe.api.riotgames.com/riot/account/v1/accounts/by-riot-id"

def fetch_puuid(game_name, tag_line, max_retries=5):
    endpoint = f"{BASE_URL}/{game_name}/{tag_line}"
    backoff_seconds = 1

    for attempt in range(1, max_retries + 1):
        try:
            print(f"[INFO] Requesting: {endpoint} (Attempt {attempt})")
            response = requests.get(endpoint, headers=HEADERS, timeout=10)

            if response.status_code == 200:
                return response.json().get("puuid")

            elif response.status_code == 429:
                wait_time = int(response.headers.get("Retry-After", backoff_seconds))
                print(f"[429] Rate limit. Waiting {wait_time}s...")
                time.sleep(wait_time)
                backoff_seconds *= 2

            elif response.status_code in [500, 503]:
                print(f"[{response.status_code}] Server error. Waiting {backoff_seconds}s...")
                time.sleep(backoff_seconds)
                backoff_seconds *= 2

            else:
                print(f"[ERROR] {response.status_code} for {game_name}#{tag_line}: {response.text}")
                return None

        except requests.exceptions.RequestException as e:
            print(f"[ERROR] Request failed for {game_name}#{tag_line}: {e}")
            time.sleep(backoff_seconds)
            backoff_seconds *= 2

    print(f"[ERROR] Exceeded max retries for {game_name}#{tag_line}.")
    return None


def load_already_done(output_csv):
    """Load already-processed Name+Tag pairs so we can skip them on resume."""
    done = set()
    if os.path.exists(output_csv):
        with open(output_csv, mode='r', encoding='utf-8') as f:
            for row in csv.DictReader(f):
                name = (row.get("Name") or "").strip()
                tag  = (row.get("Tag") or "").strip()
                if name and tag:
                    done.add((name, tag))
        print(f"[RESUME] Skipping {len(done)} already processed players.")
    return done


def main(input_csv, output_csv):
    # Load already processed players for resume support
    already_done = load_already_done(output_csv)

    # Open output file in append mode — write each row immediately
    write_header = not os.path.exists(output_csv) or os.path.getsize(output_csv) == 0
    outfile = open(output_csv, mode='a', newline='', encoding='utf-8')
    writer = csv.DictWriter(outfile, fieldnames=["Name", "Tag", "puuid"])
    if write_header:
        writer.writeheader()
        outfile.flush()

    with open(input_csv, mode='r', encoding='utf-8') as infile:
        reader = csv.DictReader(infile)

        if not {"Name", "Tag"}.issubset(reader.fieldnames or []):
            print("[ERROR] Input CSV must have headers: Name, Tag")
            outfile.close()
            sys.exit(1)

        processed = 0
        skipped   = 0

        for row in reader:
            # ── FIX 1: safely handle None values in any column ──
            name = (row.get("Name") or "").strip()
            tag  = (row.get("Tag") or "").strip()

            if not name or not tag:
                print(f"[SKIP] Missing Name or Tag in row: {row}")
                skipped += 1
                continue

            # ── FIX 2: skip already processed players (resume support) ──
            if (name, tag) in already_done:
                continue

            puuid = fetch_puuid(name, tag)

            # ── FIX 3: write immediately to disk after each player ──
            writer.writerow({
                "Name":  name,
                "Tag":   tag,
                "puuid": puuid if puuid else ""
            })
            outfile.flush()  # force write to disk right away

            processed += 1
            if processed % 100 == 0:
                print(f"[PROGRESS] Processed {processed} players so far...")

    outfile.close()
    print(f"[DONE] Processed {processed} players. Skipped {skipped} bad rows.")
    print(f"[DONE] Results saved to {output_csv}.")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python getPlayerPuuids.py <input_csv> <output_csv>")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])