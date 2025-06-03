import re
import pandas as pd

import os

input_path = os.path.join('FinalProject', 'raw_creatures.txt')
output_path = os.path.join('FinalProject','clean_creatures.csv')

def extract_first_int(s):
    """Extract the first signed integer from a string, or return None if not found."""
    match = re.search(r'-?\d+', s)
    return int(match.group()) if match else None


def clean_creature_data(input_file, output_file):
    data = []

    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            # Skip empty lines
            if not line.strip():
                continue

            parts = line.strip().split('\t')
            if len(parts) < 8:
                print(f"Skipping malformed line: {line.strip()}")
                continue

            name = parts[0]

            # Extract all numeric fields safely
            level      = extract_first_int(parts[1])
            hp_clean   = extract_first_int(parts[2])
            ac         = extract_first_int(parts[3])
            fort       = extract_first_int(parts[4])
            reflex     = extract_first_int(parts[5])
            will       = extract_first_int(parts[6])
            perception = extract_first_int(parts[7])

            # Validate that all values were found
            if None in [level, hp_clean, ac, fort, reflex, will, perception]:
                print(f"Skipping line due to missing number: {line.strip()}")
                continue

            # Append cleaned row
            data.append({
                'Name': name,
                'Level': level,
                'HP': hp_clean,
                'AC': ac,
                'Fortitude': fort,
                'Reflex': reflex,
                'Will': will,
                'Perception': perception
            })

    # Save cleaned data to CSV
    df = pd.DataFrame(data)
    df.to_csv(output_file, index=False)
    print(f"Cleaned data saved to {output_file}")



clean_creature_data(input_path, output_path)
