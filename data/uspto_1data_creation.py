"""
script created by @ZoeYou for creating the dataset for PatentMap
output: number of years of a feather file with the following columns: application_number, publication_number, decision, title, background, abstract, claims, summary, full_description, ipcr_labels


before running this script, download the data from the following link: https://huggingface.co/datasets/HUPD/hupd/resolve/main/data/all-years.tar
and uncompress the all-years.tar file, change the uncompressed directory name to 'tar_data'

"""

import os
import tarfile
import glob
import pandas as pd
import json  # Safe replacement for eval
from collections import defaultdict
from tqdm import tqdm



# Get all the tar.gz files in the directory
input_dir = './tar_data'
all_files = glob.glob(os.path.join(input_dir, "*.tar.gz"))
# Reorder the files
all_files.sort()
print("All files to be processed:", all_files)

# Create a directory to store the feather files
output_dir = "./raw_data"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Read tar.gz file contents into a dictionary
for file in all_files:
    print("Processing file:", file)
    year = file.split("/")[-1].split(".")[0]

    # Create an empty dictionary to store the data
    key_value = defaultdict(list)

    # Read the tar.gz file, convert the content into dict
    try:
        with open(file, 'rb') as f:
            with tarfile.open(fileobj=f, mode='r:gz') as tar:
                # Read the content of the tar.gz file
                for member in tqdm(tar.getmembers(), desc=f"Extracting {year}"):
                    if member.isfile():
                        try:
                            file_content = tar.extractfile(member).read().decode('utf-8')  # Decode file content as UTF-8
                            # Convert the string to dict safely
                            data = json.loads(file_content)  # Use JSON for safety
                            
                            # Validate data structure
                            if not isinstance(data, dict):
                                print(f"Warning: Invalid data structure in member {member.name}")
                                continue
                                
                            # Append the data to the dictionary
                            for key, value in data.items():
                                key_value[key].append(value)
                        except (json.JSONDecodeError, UnicodeDecodeError) as e:
                            print(f"Error processing member {member.name}: {e}")
                            continue
                        except Exception as e:
                            print(f"Unexpected error processing member {member.name}: {e}")
                            continue

    except Exception as e:
        print(f"Error processing file {file}: {e}")
        continue

    # Convert the dictionary to a DataFrame
    if not key_value:
        print(f"Warning: No valid data found for {year}. Skipping.")
        continue
        
    df = pd.DataFrame(key_value)
    
    # Validate required columns exist
    required_cols = ["application_number", "publication_number", "decision", "title", "background", "abstract", "claims", "summary", "full_description", "ipcr_labels"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"Warning: Missing columns for {year}: {missing_cols}")
        # Add missing columns with empty strings
        for col in missing_cols:
            df[col] = ""
    
    # keep only the columns we need
    df = df[required_cols]
    
    # Basic data validation
    initial_rows = len(df)
    df = df.dropna(subset=['application_number'])  # Remove rows without application number
    if len(df) < initial_rows:
        print(f"Warning: Removed {initial_rows - len(df)} rows with missing application numbers")
    
    if len(df) == 0:
        print(f"Warning: No valid data remaining for {year}. Skipping.")
        continue

    # Print DataFrame shape and preview
    print(f"Final DataFrame shape for {year}: {df.shape}")
    print(df.head())

    # Save the DataFrame to a feather file
    output_path = os.path.join(output_dir, f"dataset_{year}.feather")
    df.to_feather(output_path)
    print(f"Saved: {output_path}")
    
    # Clean up memory
    del df, key_value
    import gc
    gc.collect()

print("Processing complete!")