import os
import time
import pandas as pd
import requests


# ============================================================
# Configuration
# ============================================================

METADATA_FILE = "data/metadata/selected_subjects.csv"
OUTPUT_DIR = "data/raw"

BASE_URL = (
    "https://s3.amazonaws.com/"
    "fcp-indi/data/Projects/"
    "ABIDE_Initiative/Outputs/"
    "cpac/filt_global/alff/"
)


# ============================================================
# Create output directory
# ============================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# Load selected subjects
# ============================================================

df = pd.read_csv(METADATA_FILE)

print("=" * 60)
print("ABIDE DATA DOWNLOADER")
print("=" * 60)

print("Subjects to download:", len(df))


# ============================================================
# Download files
# ============================================================

successful = 0
failed = 0


for index, row in df.iterrows():

    file_id = row["FILE_ID"]
    label = int(row["LABEL"])

    filename = f"{file_id}_alff.nii.gz"

    url = BASE_URL + filename

    output_path = os.path.join(
        OUTPUT_DIR,
        filename
    )

    print("\n" + "-" * 60)
    print(f"[{index + 1}/{len(df)}] {file_id}")
    print("Label:", "ASD" if label == 1 else "CONTROL")
    print("URL:", url)

    # Skip if already downloaded
    if os.path.exists(output_path):

        print("Already exists. Skipping.")

        successful += 1
        continue

    try:

        response = requests.get(
            url,
            stream=True,
            timeout=60
        )

        if response.status_code == 200:

            with open(output_path, "wb") as file:

                for chunk in response.iter_content(
                    chunk_size=8192
                ):

                    if chunk:
                        file.write(chunk)

            print("Downloaded successfully.")

            successful += 1

        else:

            print(
                "Download failed."
                f" HTTP status: {response.status_code}"
            )

            failed += 1

    except Exception as e:

        print("Error:", e)

        failed += 1

    # Small delay between requests
    time.sleep(0.5)


# ============================================================
# Summary
# ============================================================

print("\n" + "=" * 60)
print("DOWNLOAD COMPLETE")
print("=" * 60)

print("Successful:", successful)
print("Failed:", failed)

print("\nFiles are stored in:")
print(OUTPUT_DIR)