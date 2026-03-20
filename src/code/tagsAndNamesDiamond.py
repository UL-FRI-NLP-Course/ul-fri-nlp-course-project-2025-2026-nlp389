import requests
from bs4 import BeautifulSoup
import csv
import time

def scrape_diamond_leaderboards(output_csv='diamond_leaderboards.csv', max_retries=3, delay=1):
    """
    Scrapes all Diamond-tier summoner names and tags from op.gg for EUW ladder.

    :param output_csv: The name/path of the CSV file to which results will be saved.
    :param max_retries: How many times to retry a request before giving up.
    :param delay: Seconds to wait between page requests to avoid hitting the server too quickly.
    """

    # base URL: includes region=euw, type=ladder, tier=diamond, page=1
    base_url = "https://www.op.gg/leaderboards/tier?region=euw&type=ladder&tier=diamond&page="

    page_number = 1
    all_entries = []  # will hold (name, tag) tuples

    while True:
        url = base_url + str(page_number)
        print(f"Scraping page {page_number} -> {url}")

        # Attempt to fetch the page with limited retries
        success = False
        for _ in range(max_retries):
            try:
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                                  'AppleWebKit/537.36 (KHTML, like Gecko) '
                                  'Chrome/108.0.0.0 Safari/537.36'
                }
                response = requests.get(url, headers=headers, timeout=10)
                if response.status_code == 200:
                    success = True
                    break
                else:
                    print(f"Received non-200 status code ({response.status_code}). Retrying...")
                    time.sleep(1)  # short wait before retry
            except requests.exceptions.RequestException as e:
                print(f"Error fetching page {page_number}: {e}. Retrying...")
                time.sleep(1)  # short wait before retry

        if not success:
            print(f"Failed to retrieve page {page_number} after {max_retries} retries. Stopping.")
            break

        soup = BeautifulSoup(response.text, 'html.parser')

        # Extract the name/tag elements (confirm these classes by inspecting the site’s HTML)
        names = soup.find_all(class_="css-ao94tw e1swkqyq1")  # Summoner names
        tags = soup.find_all(class_="css-1mbuqon e1swkqyq2")  # Summoner tags

        # If no names found, we can assume we've reached the end
        if not names or not tags:
            print("No more names/tags found. Stopping scrape.")
            break

        # Safety check in case there's a mismatch
        if len(names) != len(tags):
            print(f"Mismatch in length on page {page_number} - skipping.")
        else:
            for name, tag in zip(names, tags):
                all_entries.append((name.get_text(strip=True), tag.get_text(strip=True)))

        # Move to the next page
        page_number += 1

        # Polite delay to avoid hitting the site too fast
        time.sleep(delay)

    # Write results to CSV
    with open(output_csv, 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['Name', 'Tag'])  # header
        writer.writerows(all_entries)

    print(f"Scraped a total of {len(all_entries)} entries.")
    print(f"Data saved to {output_csv}.")

if __name__ == "__main__":
    scrape_diamond_leaderboards()


# january 1st 1735689600
# january 31th 1738367999