# extract_team_names.py

import pandas as pd
import json
import os
import sys

# --- Configuration ---
INPUT_CSV = 'data.csv'
OUTPUT_JSON = 'valid_teams.json'


# ---------------------

def main():
    """
    Reads the main data.csv file, extracts all unique team names,
    and saves them to a JSON file for model validation.
    """
    print("--- Team Name Extractor ---")

    # 1. Check if the source data file exists
    if not os.path.exists(INPUT_CSV):
        print(f"\n❌ ERROR: Input file '{INPUT_CSV}' not found.")
        print("Please ensure you have run your scraping scripts to generate the data file.")
        sys.exit(1)  # Exit the script with an error code

    # 2. Load the data using pandas (efficiently)
    print(f"Reading team data from '{INPUT_CSV}'...")
    try:
        # We only need the team name columns, which saves memory on large files
        df = pd.read_csv(INPUT_CSV, usecols=['team_a', 'team_b'])
        print(f"Successfully loaded {len(df)} rows.")
    except Exception as e:
        print(f"\n❌ ERROR: Failed to read the CSV file. Reason: {e}")
        sys.exit(1)

    # 3. Extract unique names
    print("Extracting unique team names...")

    # Concatenate the 'team_a' and 'team_b' columns into a single series,
    # drop any potential missing values, and then get the unique names.
    # This is a very efficient one-liner in pandas.
    all_names = pd.concat([df['team_a'], df['team_b']]).dropna().unique()

    # 4. Sort the list of names alphabetically
    sorted_teams = sorted(list(all_names))

    found_count = len(sorted_teams)
    if found_count == 0:
        print("\n⚠️ Warning: No team names were found in the data file.")
        return

    print(f"Found {found_count} unique team names.")

    # 5. Save the list to the output JSON file
    print(f"Saving the list to '{OUTPUT_JSON}'...")
    try:
        with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
            # indent=4 makes the JSON file readable
            # ensure_ascii=False correctly handles names with special characters (like KRÜ)
            json.dump(sorted_teams, f, indent=4, ensure_ascii=False)
    except Exception as e:
        print(f"\n❌ ERROR: Failed to write to the JSON file. Reason: {e}")
        sys.exit(1)

    print(f"\n✅ Success! The file '{OUTPUT_JSON}' has been created.")
    print("You can now use this file in 'main.py' to validate team inputs.")


if __name__ == "__main__":
    main()