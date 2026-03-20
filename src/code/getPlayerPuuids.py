import os
import sys
import csv
import time
import requests

# Load API Key from environment variable
API_KEY = os.getenv("RIOT_API_KEY")

if not API_KEY:
    raise ValueError("API key not found. Please set the RIOT_API_KEY environment variable.")

HEADERS = {"X-Riot-Token": API_KEY}
BASE_URL = "https://europe.api.riotgames.com/riot/account/v1/accounts/by-riot-id"

def fetch_puuid(game_name, tag_line, max_retries=5):
    """
    Fetch the PUUID for the provided Name/Tag from the Riot API.
    Retries up to `max_retries` times if we encounter rate limits (429)
    or server errors (500/503). Implements a combination of honoring the
    Retry-After header and exponential backoff.
    """
    endpoint = f"{BASE_URL}/{game_name}/{tag_line}"
    backoff_seconds = 1  # Initial backoff time if no Retry-After is given

    for attempt in range(1, max_retries + 1):
        try:
            print(f"[INFO] Requesting: {endpoint} (Attempt {attempt})")
            response = requests.get(endpoint, headers=HEADERS)

            if response.status_code == 200:
                # Success; parse and return the PUUID
                data = response.json()
                return data.get("puuid")

            elif response.status_code == 429:
                # Rate-limit exceeded
                retry_after = response.headers.get("Retry-After")
                if retry_after:
                    # Honor the server's suggested wait time
                    wait_time = int(retry_after)
                    print(f"[429] Rate limit exceeded. Retry-After: {wait_time}s")
                else:
                    # If Retry-After is not provided, do an exponential backoff
                    wait_time = backoff_seconds
                    print(f"[429] Rate limit exceeded, no Retry-After. Backing off for {wait_time}s.")
                    backoff_seconds *= 2  # Exponential factor

                time.sleep(wait_time)
                continue  # Then retry

            elif response.status_code in [500, 503]:
                # Server or service unavailable errors, might be transient
                print(f"[{response.status_code}] Server/Service error. Backoff for {backoff_seconds}s.")
                time.sleep(backoff_seconds)
                backoff_seconds *= 2
                continue  # Retry

            else:
                # Other error codes (400, 403, 404, etc.) => no point retrying
                print(f"[ERROR] {response.status_code} for {game_name}#{tag_line}: {response.text}")
                return None

        except requests.exceptions.RequestException as e:
            # Network issues or other unexpected exceptions
            print(f"[ERROR] Request failed for {game_name}#{tag_line}: {e}")
            # Exponential backoff for subsequent attempts
            time.sleep(backoff_seconds)
            backoff_seconds *= 2

    # If we exhaust all retries, return None
    print(f"[ERROR] Exceeded maximum retries ({max_retries}) for {game_name}#{tag_line}.")
    return None


def main(input_csv, output_csv):
    """
    1. Read Name/Tag from `input_csv`.
    2. For each, call the Riot endpoint to get the puuid.
    3. Write a CSV of Name, Tag, and puuid to `output_csv`.
    """
    results = []

    # Read from the input CSV
    with open(input_csv, mode='r', encoding='utf-8') as infile:
        reader = csv.DictReader(infile)
        # Ensure we have the columns we expect
        if not {"Name", "Tag"}.issubset(reader.fieldnames or []):
            print("[ERROR] Input CSV must have headers: Name, Tag")
            sys.exit(1)

        for row in reader:
            game_name = row["Name"].strip()
            tag_line = row["Tag"].strip()

            if not game_name or not tag_line:
                print(f"[SKIP] Invalid row - missing Name or Tag. Row: {row}")
                continue

            puuid = fetch_puuid(game_name, tag_line)
            # Even if puuid is None, we store something so we have a record
            results.append({
                "Name": game_name,
                "Tag": tag_line,
                "puuid": puuid if puuid else ""
            })

    # Write to the output CSV
    write_header = not os.path.exists(output_csv)  # Only write header if file doesn't exist
    with open(output_csv, mode='a', newline='', encoding='utf-8') as outfile:
        fieldnames = ["Name", "Tag", "puuid"]
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)

        if write_header:
            writer.writeheader()

        for row in results:
            writer.writerow(row)

    print(f"[DONE] PUUIDs saved to {output_csv}.")


if __name__ == "__main__":
    """
    Usage:
      python get_puuids_by_riotid.py <input_csv> <output_csv>

    <input_csv> should have columns: Name, Tag
    <output_csv> will store: Name, Tag, puuid
    """
    if len(sys.argv) != 3:
        print("Usage: python get_puuids_by_riotid.py <input_csv> <output_csv>")
        sys.exit(1)

    input_csv = sys.argv[1]
    output_csv = sys.argv[2]
    main(input_csv, output_csv)
