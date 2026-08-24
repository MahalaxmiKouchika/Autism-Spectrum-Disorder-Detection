import os
import pandas as pd


# ============================================================
# Configuration
# ============================================================

METADATA_PATH = "data/metadata/phenotypic.csv"

# Number of subjects to use initially
# Keep this small for the first prototype.
N_ASD = 20
N_CONTROL = 20


# ============================================================
# Load ABIDE Metadata
# ============================================================

def load_metadata(csv_path=METADATA_PATH):

    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"Metadata file not found: {csv_path}"
        )

    df = pd.read_csv(csv_path)

    print("=" * 60)
    print("ABIDE METADATA LOADED")
    print("=" * 60)

    print("Total subjects:", len(df))
    print("Total columns:", len(df.columns))

    return df


# ============================================================
# Clean and Prepare Labels
# ============================================================

def prepare_labels(df):

    # Keep only ASD and Control subjects
    df = df[
        df["DX_GROUP"].isin([1, 2])
    ].copy()

    # Convert ABIDE labels:
    #
    # DX_GROUP = 1 → ASD → 1
    # DX_GROUP = 2 → Control → 0

    df["LABEL"] = df["DX_GROUP"].map({
        1: 1,
        2: 0
    })

    return df


# ============================================================
# Select Balanced Dataset
# ============================================================

def select_subjects(
    df,
    n_asd=N_ASD,
    n_control=N_CONTROL
):

    # ASD
    asd = df[
        df["LABEL"] == 1
    ].copy()

    # Control
    control = df[
        df["LABEL"] == 0
    ].copy()

    print("\nAvailable subjects:")
    print("ASD:", len(asd))
    print("CONTROL:", len(control))

    # Select subjects
    asd = asd.head(n_asd)

    control = control.head(n_control)

    # Combine
    selected = pd.concat(
        [asd, control],
        ignore_index=True
    )

    return selected


# ============================================================
# Display Selected Subjects
# ============================================================

def display_subjects(df):

    print("\n" + "=" * 60)
    print("SELECTED SUBJECTS")
    print("=" * 60)

    columns = [
        "FILE_ID",
        "SITE_ID",
        "SUB_ID",
        "DX_GROUP",
        "LABEL"
    ]

    available_columns = [
        column
        for column in columns
        if column in df.columns
    ]

    print(
        df[available_columns].to_string(
            index=False
        )
    )

    print("\nTotal selected:", len(df))

    print(
        "ASD:",
        len(df[df["LABEL"] == 1])
    )

    print(
        "CONTROL:",
        len(df[df["LABEL"] == 0])
    )


# ============================================================
# Create Download Information
# ============================================================

def create_download_info(df):

    records = []

    for _, row in df.iterrows():

        file_id = row["FILE_ID"]

        label = int(row["LABEL"])

        # C-PAC + filt_global + ALFF
        #
        # We will use this later for downloading
        # the imaging data.

        url = (
            "https://s3.amazonaws.com/"
            "fcp-indi/data/Projects/"
            "ABIDE_Initiative/Outputs/"
            "cpac/filt_global/alff/"
            f"{file_id}_alff.nii.gz"
        )

        records.append({
            "FILE_ID": file_id,
            "LABEL": label,
            "URL": url
        })

    return pd.DataFrame(records)


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    # 1. Load metadata
    df = load_metadata()

    # 2. Prepare labels
    df = prepare_labels(df)

    # 3. Select balanced subjects
    selected = select_subjects(
        df,
        n_asd=N_ASD,
        n_control=N_CONTROL
    )

    # 4. Display subjects
    display_subjects(selected)

    # 5. Create download information
    download_df = create_download_info(
        selected
    )

    # 6. Save selected subjects
    output_path = (
        "data/metadata/"
        "selected_subjects.csv"
    )

    download_df.to_csv(
        output_path,
        index=False
    )

    print(
        "\nDownload information saved to:"
    )

    print(output_path)