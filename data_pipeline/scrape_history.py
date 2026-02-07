import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import os
import re

BASE_URL = "https://world-nuclear.org"
MAIN_URL = "https://world-nuclear.org/information-library/facts-and-figures/world-nuclear-power-reactors-and-uranium-requireme"
# Output to central data folder
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')

def get_soup(url):
    try:
        # random user agent to avoid blocking
        headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.114 Safari/537.36'
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        return BeautifulSoup(response.content, 'html.parser')
    except Exception as e:
        print(f"Error fetching {url}: {e}")
        return None

def extract_world_total_bs4(table):
    try:
        rows = table.find_all('tr')
        for tr in rows:
            # Check text content for WORLD
            if "WORLD" in tr.get_text().upper():
                # Found row
                cells = tr.find_all(['td', 'th'])
                if not cells:
                    continue
                
                # Iterate backwards to find the large number (Uranium demand)
                for cell in reversed(cells):
                    txt = cell.get_text(strip=True)
                    # Clean cleaning
                    clean = re.sub(r'[^\d]', '', txt)
                    if clean.isdigit():
                        val = int(clean)
                        # Sanity check: Global demand is typically 40k - 70k tonnes
                        if val > 40000 and val < 100000:
                            return val
    except Exception as e:
        print(f"  BS4 extraction error: {e}")
    return None

def parse_table_from_url(url, date_str):
    print(f"  Parsing table from {url} ({date_str})...")
    soup = get_soup(url)
    if not soup:
        print(f"  Failed to get soup for {url}")
        return None
        
    # Find table
    tables = soup.find_all('table')
    print(f"  Found {len(tables)} tables on page.")
    target_table = None
    
    # Heuristic: look for 'Country' in table
    for i, table in enumerate(tables):
        txt = table.get_text()
        if 'Country' in txt or 'Operating' in txt:
            # print(f"  Table {i} matches keywords.")
            target_table = table
            break
    
    if not target_table and tables:
        print("  No keyword match, using Table 0.")
        target_table = tables[0]
        
    if not target_table:
        print(f"  No table found for {date_str} (tables list empty).")
        return None
        
    try:
        # Use header=0 to try to auto-detect header? No, header=None is safer.
        dfs = pd.read_html(str(target_table), header=None)
        if not dfs:
            print("  pd.read_html returned empty list.")
            return None
        df = dfs[0]
        # print(f"  DataFrame extracted: {df.shape}")
        
        df['Date'] = date_str
        
        # DEBUG: Print tail to see where WORLD is
        # print("DataFrame Tail:")
        # print(df.tail(10))
        
        # Check col 0
        # print("Col 0 tail:")
        # print(df.iloc[:,0].tail(10).values)

        return df
        
    except Exception as e:
        print(f"  Error parsing table content: {e}")
        return None

def scrape_history():
    print(f"Fetching main page: {MAIN_URL}")
    soup = get_soup(MAIN_URL)
    if not soup:
        return

    # Find the "Earlier tables" section
    links = []
    
    # Regex to find Month Year format
    date_pattern = re.compile(r'(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}')
    
    # Find all links in the page
    all_links = soup.find_all('a', href=True)
    
    for a in all_links:
        text = a.get_text().strip()
        if date_pattern.match(text):
            href = a['href']
            if not href.startswith('http'):
                href = BASE_URL + href
            links.append({'date': text, 'url': href})
            
    print(f"Found {len(links)} historical links.")
    
    all_dfs = []
    
    # Re-impl loop with refactored logic
    for i, item in enumerate(links):
        print(f"[{i+1}/{len(links)}] Processing {item['date']}...")
        soup = get_soup(item['url'])
        if not soup:
            continue
            
        # Find table
        tables = soup.find_all('table')
        target_table = None
        for table in tables:
            txt = table.get_text()
            if 'Country' in txt or 'Operating' in txt:
                target_table = table
                break
        if not target_table and tables:
            target_table = tables[0]
            
        if target_table:
            # 1. Try to get efficient Global Demand via BS4
            total_u = extract_world_total_bs4(target_table)
            if total_u:
                print(f"  > Extracted World Demand: {total_u}")
                all_dfs.append({'Date': item['date'], 'Global_Uranium_Demand': total_u})
            else:
                print(f"  > Could not extract world total via BS4")
                
            # 2. Save CSV via pandas (optional but good)
            try:
                dfs = pd.read_html(str(target_table), header=None)
                if dfs:
                    df = dfs[0]
                    safe_date = item['date'].replace(' ', '_')
                    save_path = f"{OUTPUT_DIR}/history/uranium_{safe_date}.csv"
                    os.makedirs(os.path.dirname(save_path), exist_ok=True)
                    df.to_csv(save_path, index=False, header=False)
            except:
                pass
                
        time.sleep(1)

    if all_dfs:
        final_df = pd.DataFrame(all_dfs)
        # Convert Date to datetime
        final_df['Date'] = pd.to_datetime(final_df['Date'])
        final_df = final_df.sort_values('Date')
        
        path = f"{OUTPUT_DIR}/uranium_time_series.csv"
        final_df.to_csv(path, index=False)
        print(f"\nSaved time-series data to {path}")
        print(final_df.head())
        print(final_df.tail())
    else:
        print("No data extracted.")

if __name__ == "__main__":
    scrape_history()
