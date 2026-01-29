# KNIME Python Script: Clean b_Score Column
# Purpose: Remove single quotes from the b_Score column
# Input: Single table with b_Score column
# Output: Same table with cleaned b_Score column

import knime.scripting.io as knio

# Read input table
df = knio.input_tables[0].to_pandas()

# Check if b_Score column exists
if 'b_Score' not in df.columns:
    raise ValueError("Column 'b_Score' not found in input table")

# Remove single quotes from b_Score column
df['b_Score'] = df['b_Score'].astype(str).str.replace("'", "", regex=False)

# Output the cleaned DataFrame
knio.output_tables[0] = knio.Table.from_pandas(df)
