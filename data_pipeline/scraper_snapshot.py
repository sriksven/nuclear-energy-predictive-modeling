import requests
from bs4 import BeautifulSoup
import pandas as pd
import io
import os

def scrape_data():
    url = "https://world-nuclear.org/information-library/facts-and-figures/world-nuclear-power-reactors-archive/world-nuclear-power-reactors-2005-07-and-urani-(1)"
    
    print(f"Fetching {url}...")
    try:
        response = requests.get(url)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching URL: {e}")
        return

    soup = BeautifulSoup(response.content, 'html.parser')
    
    # Strategy: Find all tables and look for one with specific headers or keywords
    tables = soup.find_all('table')
    print(f"Found {len(tables)} tables.")
    
    target_table = None
    
    if not target_table:
        if len(tables) > 0:
            print("Keyword check failed, but tables found. Using the first table.")
            target_table = tables[0]
        else:
            print("No tables found.")
            return

    # Parse table using pandas
    try:
        # Use header=None to see purely what's there
        dfs = pd.read_html(str(target_table), header=None)
        if dfs:
            df = dfs[0]
            
            # The last few rows are headers/metadata, and "WORLD**" is the last data row.
            # Based on inspection:
            # Row 0 to 38 are countries (inc World at 38).
            # Row 39, 40 are headers.
            
            # Let's slice off the non-country rows. 
            # We iterate to find where "WORLD" or headers start to be safe, or just hardcode based on inspection but dynamic is better.
            # However, looking at the previous output, "WORLD**" was row 38 (line 39 in file).
            # And then garbage/headers.
            
            print("DataFrame columns:", df.columns)
            
            # Find the index of the row starting with "WORLD"
            # Use iloc[:, 0] to safely access the first column regardless of column names
            world_index = df[df.iloc[:, 0].astype(str).str.contains("WORLD", case=False, na=False)].index
            
            if not world_index.empty:
                last_data_index = world_index[0]
                df_clean = df.iloc[:last_data_index].copy()
            else:
                df_clean = df.copy()
            
            # Define columns
            cols = [
                "Country", 
                "Nuclear_Electricity_2005_billion_kWh", 
                "Nuclear_Electricity_2005_percent", 
                "Reactors_Operating_No", 
                "Reactors_Operating_MWe", 
                "Reactors_Building_No", 
                "Reactors_Building_MWe", 
                "Reactors_Planned_No", 
                "Reactors_Planned_MWe", 
                "Reactors_Proposed_No", 
                "Reactors_Proposed_MWe", 
                "Uranium_Required_2007_tonnes"
            ]
            
            # Reset columns to simple validation
            if len(df_clean.columns) == len(cols):
                df_clean.columns = cols
            else:
                 print(f"Warning: Column count mismatch. Expected {len(cols)}, got {len(df_clean.columns)}")
            
            # Clean Country column (remove *)
            # Use iloc[:, 0] if column assignment failed, or use 'Country' if it worked
            if 'Country' in df_clean.columns:
                df_clean['Country'] = df_clean['Country'].astype(str).str.replace('*', '', regex=False).str.strip()
            else:
                # If column names match fail, operate on first column by position
                df_clean.iloc[:, 0] = df_clean.iloc[:, 0].astype(str).str.replace('*', '', regex=False).str.strip()
            
            # Clean numeric columns
            # If columns assigned
            if 'Country' in df_clean.columns:
                for col in cols[1:]:
                    df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce').fillna(0)
            else:
                 print("Skipping numeric conversion due to column mismatch.")

            print("\nCleaned DataFrame head:")
            print(df_clean.head())
            print(f"\nDimensions: {df_clean.shape}")
            
            # Updated path for new structure
            output_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'uranium_snapshot.csv')
            df_clean.to_csv(output_path, index=False)
            print(f"Saved final data to {output_path}")
            
    except Exception as e:
        print(f"Error parsing table with pandas: {e}")

if __name__ == "__main__":
    scrape_data()
