"""
script created by @ZoeYou for creating the dataset for PatentMap
output: number of years of a feather file with the following columns: abstract, claims, summary, description
"""

import pandas as pd
import os
import glob
import re
import gc
import argparse
from tqdm import trange

from collections import defaultdict, OrderedDict



def reorganize_claim_set(claims):
    """remove cancelled claims at the beginning of the claim set"""
    cancelled_head = re.compile(r"(\d+-\d+|\d+)\.\s*\((cancelled|canceled)\)\s*")

    # check if the claim set is empty
    if not claims:
        return ""
    else:
        # check if the claim set has "(cancelled)" in the text
        if re.search(cancelled_head, claims):
            # remove the cancelled claims at the beginning of the claim set
            claims = re.sub(cancelled_head, "", claims)
        return claims


UPPER_HEADING_REGEX = re.compile(r"((?:[A-Z]{2,}\s){2,}|(?:[A-Z]{4,}\s){1,})")
def detect_headings(text):
    """
    Detects headings full_description based on all-uppercase expressions.
    """
    matches = [match.strip() for match in UPPER_HEADING_REGEX.findall(text)]

    # return all matches, remove duplicates without changing order
    matches = list(OrderedDict.fromkeys(matches))
    return matches


def extract_drawing_and_detailed_description(full_description):
    """
    Extracts the drawing and detailed description from the full_description.
    """
    #  match all uppercase headings with position information
    headings, heading_positions = [], []
    for match in UPPER_HEADING_REGEX.finditer(full_description):
        headings.append(match.group().strip())
        heading_positions.append((match.start(), match.end()))

    # get the sections based on the positions of the headings
    sections = []
    for i in range(len(headings)):
        if i == 0:
            if heading_positions[i][0] != 0:
                sections.append(full_description[:heading_positions[i][0]].strip())
        else:
            sections.append(full_description[heading_positions[i-1][1]:heading_positions[i][0]].strip())
    # add the last section
    if heading_positions:   sections.append(full_description[heading_positions[-1][1]:].strip())
    else:   sections.append(full_description.strip())

    # remove empty strings from sections
    assert len(sections) == len(headings) or len(sections) == len(headings) + 1, f"The number of sections and headings do not match, {len(sections)} != {len(headings)}"

    # find the index of the drawing section
    drawing_index = next((i for i, heading in enumerate(headings) if re.search(r"DRAWING|FIGURE", heading, re.IGNORECASE)), None)
    if drawing_index is not None:
        # extract the drawing section
        drawing_section = sections[drawing_index]
        if len(drawing_section.split(" ")) < 15:
            # if the drawing section is too short, it is likely not a drawing
            # so we will not extract it
            drawing_section = ""
    else:
        drawing_section = ""

    # extract the detailed description section (the section with "DETAILED" in its heading)
    detailed_description_index = next((i for i, heading in enumerate(headings) if re.search(r"DETAILED|EMBODIMENT", heading, re.IGNORECASE)), None)
    if detailed_description_index is not None:
        detailed_description_section = sections[detailed_description_index]
        if len(detailed_description_section.split(" ")) < 15:
            # if the detailed description section is too short, it is likely not a detailed description
            # so we will not extract it
            detailed_description_section = ""
    else:
        detailed_description_section = ""

    # return the drawing section, detailed description section, and the original full_description
    return drawing_section, detailed_description_section


def safe_post_eoh_extraction(x):
    """Safely extract summary and background information by removing the heading and trailing spaces."""
    if '<EOH>' in x:
        parts = x.split('<EOH>', 1)  # Split only once explicitly
        return parts[1].strip() if parts[1].strip() else ""
    return x.strip()


def is_summary_drawing(summary):
    """Check if the summary is actually a drawing description."""
    drawing_pattern = re.compile(r' (?:drawing|figure)s?', re.IGNORECASE)
    if '<EOH>' in summary and '<SOH>' in summary:
        heading = summary.split('<EOH>')[0]
        if drawing_pattern.search(heading):
            return ""
        
    # remove the strange tail such as 'detailed-description description="Detailed Description" end="lead"?"' and 'BRFSUM description="Brief Summary" end="tail"?"'
    summary = re.sub(r' (?:detailed-description|brfsum) description=".*?" end="(?:lead|tail)"\?', '', summary)
    return summary.strip()


def count_number_of_words(df, target_cols):
    """Count the average number of words in the target columns."""
    for col in target_cols:
        non_empty_avg = df[df[f'{col}_word_count'] > 0][f'{col}_word_count'].mean()
        if non_empty_avg == 0:
            raise ValueError(f"All values in {col} are empty or contain only whitespace.")
        
        print(f"Average number of words in {col}: {non_empty_avg:.2f}")
    print("\n")


def truncate_text(text, max_words=1000):
    if not isinstance(text, str):
        return ""
    return ' '.join(text.split()[:max_words])


def normalize_whitespace(text):
    if not isinstance(text, str):
        return ""
    return re.sub(r'\s+', ' ', text).strip()


def process_file_batch(df_batch):
    """Process a batch of the dataframe safely."""
    if df_batch.empty:
        return df_batch
        
    df_batch = df_batch.copy()

    # Fill NaN values with empty strings before processing
    text_cols = ["abstract", "claims", "summary", "background", "full_description", "title"]
    for col in text_cols:
        if col in df_batch.columns:
            df_batch[col] = df_batch[col].fillna("")

    # Normalize whitespace
    for col in ["abstract", "claims", "summary", "background", "full_description"]:
        if col in df_batch.columns:
            df_batch[col] = df_batch[col].apply(normalize_whitespace)
    
    if "claims" in df_batch.columns:
        df_batch["claims"] = df_batch["claims"].apply(reorganize_claim_set)

    # extract drawing and detailed description
    df_batch["drawing"], df_batch["detailed_description"] = zip(
        *df_batch["full_description"].apply(extract_drawing_and_detailed_description)
    )

    # empty summaries that are actually drawing descriptions and not summaries
    df_batch["summary"] = df_batch["summary"].apply(is_summary_drawing)

    # remove headings from summary and background
    df_batch["summary"] = df_batch["summary"].apply(safe_post_eoh_extraction)
    df_batch["background"] = df_batch["background"].apply(safe_post_eoh_extraction) 

    # truncate early to speed processing significantly
    for col in ["title", "abstract", "claims", "summary", "background", "full_description", "drawing", "detailed_description"]:
        # Truncate text to a maximum of 1000 words
        df_batch[f"{col}_word_count"] = df_batch[col].apply(lambda x: len(x.split()))
        df_batch[col] = df_batch[col].apply(lambda x: truncate_text(x, 1000))

    return df_batch


def process_in_batches(feather_files, target_cols, batch_size=2048, temp_dir="./temp_batches"):
    os.makedirs(temp_dir, exist_ok=True)
    temp_files = []
    batch_id = 0

    for file in feather_files:
        print(f"Processing {file}...")
        df = pd.read_feather(file, columns=target_cols)
        print(f"{file} Memory usage:", df.memory_usage(deep=True).sum() / (1024**3), "GB")

        num_rows = len(df)
        for start in trange(0, num_rows, batch_size, desc=f"Processing batches in {file}"):
            batch = df.iloc[start:start + batch_size]
            processed_batch = process_file_batch(batch)

            # Write batch to disk
            temp_path = os.path.join(temp_dir, f"batch_{batch_id}.feather")
            processed_batch.reset_index(drop=True).to_feather(temp_path)
            temp_files.append(temp_path)
            batch_id += 1

            del batch, processed_batch
            gc.collect()

        del df
        gc.collect()

    # Load and concat only once at the end
    df_final = pd.concat((pd.read_feather(f) for f in temp_files), ignore_index=True)

    # Clean up temp files
    for f in temp_files:
        os.remove(f)
    
    # remove temp directory
    os.rmdir(temp_dir)

    return df_final


def analyze_full_description(list_of_matches):
    """
    Analyzes the matches of uppercase sections in the full_description.

    """
    # count the occurrences of each pattern (order of sections)
    pattern_counts = defaultdict(int)
    for match in list_of_matches:
        pattern_counts["-".join(match)] += 1
    # sort the patterns by their counts
    sorted_patterns = sorted(pattern_counts.items(), key=lambda x: x[1], reverse=True)

    # print the 25 most common pattern
    print("Most common patterns in full_description:")
    for pattern, count in sorted_patterns[:25]:
        print(f"{pattern} : {count}")
    print("=====================================================")

    # print the most common pattern
    most_common_pattern = max(pattern_counts, key=pattern_counts.get)
    print(f"Most common pattern: {most_common_pattern} with {pattern_counts[most_common_pattern]} occurrences")
    print("=====================================================")



def main():
    parser = argparse.ArgumentParser(description="Process patent documents.")
    parser.add_argument("--input", type=str, default='./raw_data', help="Input directory of raw feather files")
    parser.add_argument("--output", type=str, default='./preprocessed_data', help="Output directory for preprocessed files")
    parser.add_argument("--start_year", type=int, default=2004, help="Start year for filtering")
    parser.add_argument("--end_year", type=int, default=2018, help="End year for filtering")

    args = parser.parse_args()

    print("Input folder:", args.input)
    print("Output file:", args.output)

    os.makedirs(args.output, exist_ok=True)

    target_cols = [
        'title', 'abstract', 'claims', 'summary', 'background', 'full_description',
        'application_number',  'ipcr_labels'
    ]

    global_word_counts = defaultdict(int)
    global_doc_count = 0
   
    description_patterns = []
    application_numbers = []
    for year in range(args.start_year, args.end_year + 1):
        print(f"\n=== Processing year {year} ===")

        feather_files = sorted(
            file for file in glob.glob(os.path.join(args.input, "*.feather"))
            if re.search(rf"{year}", os.path.basename(file))
        )

        if not feather_files:
            print(f"No files found for year {year}. Skipping.")
            continue

        df_final = process_in_batches(feather_files, target_cols)

        # analyze the pattern of the full description
        description_patterns.extend(df_final["full_description"].astype(str).apply(detect_headings).tolist())

        application_numbers.extend(df_final["application_number"].tolist())

        # Drop duplicates and reset indexc (duplicates are counted based on the hash of the text)
        df_final["row_hash"] = pd.util.hash_pandas_object(
            df_final[["abstract", "claims", "summary", "background", "full_description"]], index=True
        ).astype(str)

        df_final = df_final.drop_duplicates(subset=["row_hash"]).drop(columns=["row_hash"]).reset_index(drop=True)
        print(f"Final dataset shape for {year}: {df_final.shape}")

        # Count the average number of words in each column
        print("Average number of words in each column:")
        count_number_of_words(df_final, ["title", "abstract", "claims", "summary", "background", "full_description", "drawing", "detailed_description"])

        # Update global stats
        for col in ["title", "abstract", "claims", "summary", "background", "full_description", "drawing", "detailed_description"]:
            # Update global word counts
            global_word_counts[col] += df_final[df_final[f"{col}_word_count"] > 0][f"{col}_word_count"].sum()

        global_doc_count += len(df_final)

        # rename columns
        renames = {"claims": "claim", "full_description": "invention"}
        df_final.rename(columns=renames, inplace=True)

        # Print samples
        print("Random abstracts:")
        print(df_final["abstract"].sample(n=min(3, len(df_final))).values)
        print("--" * 50)
        print("Random summaries:")
        print(df_final["summary"].sample(n=min(3, len(df_final))).values)
        print("--" * 50)
        print("Random backgrounds:")
        print(df_final["background"].sample(n=min(3, len(df_final))).values)
        print("--" * 50)
        print("Random claims:")
        print(df_final["claim"].sample(n=min(3, len(df_final))).values)
        print("--" * 50)
        print("Random figures:")
        print(df_final["drawing"].sample(n=min(3, len(df_final))).values)
        print("--" * 50)
        print("Random detailed descriptions:")
        print(df_final["detailed_description"].sample(n=min(3, len(df_final))).values)
        print("--" * 50)

        # Save output
        out_path = os.path.join(args.output, f"patentmap_dataset_{year}.feather")
        df_final.reset_index(drop=True).to_feather(out_path)

        # Clean up
        del df_final
        gc.collect()
        print(f"Saved preprocessed file for {year}: {out_path}")

    # # analyze the full description
    # analyze_full_description(description_patterns)

    # check if there is any duplicate application numbers
    application_numbers = set(application_numbers)
    all_application_numbers = application_numbers
    print("Total application numbers:", len(all_application_numbers))
    print("Unique application numbers:", len(set(all_application_numbers)))


    print("\n=== Global Statistics ===")
    print(f"Total number of documents: {global_doc_count}")
    for col in ["title", "abstract", "claims", "background", "summary", "drawing", "detailed_description"]:
        avg_words = global_word_counts[col] / global_doc_count if global_doc_count else 0
        print(f"Average number of words in {col}: {avg_words:.2f}")


if __name__ == "__main__":
    main()
