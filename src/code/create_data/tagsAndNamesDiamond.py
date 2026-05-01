import requests
from bs4 import BeautifulSoup
import csv
import json
import time
import re
from urllib.parse import unquote


def scrape_diamond_leaderboards(
    output_csv='diamond_leaderboards.csv',
    max_page=237,
    delay=1.5,
    max_retries=3
):
    base_url = "https://www.op.gg/leaderboards/tier?region=euw&type=ladder&tier=diamond&page="
    all_entries = []

    for page_number in range(1, max_page + 1):
        url = base_url + str(page_number)
        print(f"Scraping page {page_number}/{max_page} -> {url}")

        response = None
        for attempt in range(max_retries):
            try:
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                                  'AppleWebKit/537.36 (KHTML, like Gecko) '
                                  'Chrome/124.0.0.0 Safari/537.36',
                    'Accept-Language': 'en-US,en;q=0.9',
                }
                response = requests.get(url, headers=headers, timeout=10)
                if response.status_code == 200:
                    break
                print(f"  Status {response.status_code}, retry {attempt+1}...")
                time.sleep(2)
            except Exception as e:
                print(f"  Error: {e}, retry {attempt+1}...")
                time.sleep(2)
        else:
            print(f"  Failed after {max_retries} retries. Skipping page.")
            continue

        soup = BeautifulSoup(response.text, 'html.parser')

        # ── METHOD 1: __NEXT_DATA__ JSON ──
        next_data_tag = soup.find('script', id='__NEXT_DATA__')
        if next_data_tag:
            try:
                data = json.loads(next_data_tag.string)
                players = (
                    data.get('props', {})
                        .get('pageProps', {})
                        .get('data', [])
                )
                page_entries = []
                for player in players:
                    summoner = player.get('summoner', player)
                    name = summoner.get('game_name') or summoner.get('name', '')
                    tag  = summoner.get('tag_line')  or summoner.get('tag', '')
                    if name and tag:
                        page_entries.append((name, tag))

                if page_entries:
                    print(f"  ✓ Found {len(page_entries)} players via __NEXT_DATA__")
                    all_entries.extend(page_entries)
                    time.sleep(delay)
                    continue

                print("  __NEXT_DATA__ found but no players parsed, trying fallback...")

            except (json.JSONDecodeError, AttributeError) as e:
                print(f"  JSON parse error: {e}, trying fallback...")

        # ── METHOD 2: Table row fallback ──
        tbody = soup.find('tbody')
        if not tbody:
            print("  No <tbody> found. Likely JS-rendered. Stopping.")
            break

        page_entries = []
        for row in tbody.find_all('tr'):
            anchor = row.find('a', href=True)
            if anchor and '/summoners/' in anchor['href']:
                slug = unquote(anchor['href'].split('/')[-1])
                if '-' in slug:
                    parts = slug.rsplit('-', 1)
                    name = parts[0]
                    tag  = parts[1]
                    page_entries.append((name, tag))
                    continue

            # Regex fallback for Name#TAG pattern
            match = re.search(r'([^|#\n]{2,20})#([A-Z0-9]{3,6})', row.get_text('|'))
            if match:
                page_entries.append((match.group(1).strip(), match.group(2).strip()))

        if not page_entries:
            print("  No entries found via fallback. Stopping.")
            break

        print(f"  ✓ Found {len(page_entries)} players via table fallback")
        all_entries.extend(page_entries)
        time.sleep(delay)

    # Save to CSV
    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Name', 'Tag'])
        writer.writerows(all_entries)

    print(f"\nDone. {len(all_entries)} entries saved to {output_csv}.")


if __name__ == "__main__":
    scrape_diamond_leaderboards()