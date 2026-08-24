import os
import cv2
import numpy as np
import pandas as pd
import nibabel as nib
from nilearn.image import resample_img

RAW_DIR = "data/raw"
OUTPUT_DIR = "data/processed"
METADATA = "data/metadata/selected_subjects.csv"


def preprocess_image(image):
    """
    Preprocess a 2D image for the CNN prototype.
    Resizes to 128x128 and normalizes to [0, 1].
    """
    if len(image.shape) > 2:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
    resized = cv2.resize(image, (128, 128))
    normalized = resized.astype(np.float32) / 255.0
    return normalized


def load_image_dataset(data_path):
    """
    Loads 2D image dataset for the CNN prototype.
    Generates dummy data if directories are empty/missing.
    """
    X, y = [], []
    
    classes = {"CONTROL": 0, "ASD": 1}
    loaded_real_data = False
    
    for class_name, label in classes.items():
        class_dir = os.path.join(data_path, class_name)
        if os.path.exists(class_dir):
            for filename in os.listdir(class_dir):
                if filename.endswith((".png", ".jpg", ".jpeg")):
                    img_path = os.path.join(class_dir, filename)
                    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                    if img is not None:
                        processed = preprocess_image(img)
                        X.append(processed)
                        y.append(label)
                        loaded_real_data = True
                        
    if not loaded_real_data:
        print(f"Warning: No real images found in {data_path}. Generating dummy data for prototype demo...")
        for i in range(100):
            dummy_img = np.random.randint(0, 255, (256, 256), dtype=np.uint8)
            processed = preprocess_image(dummy_img)
            X.append(processed)
            y.append(i % 2)
            
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int32)
    
    X = X[..., np.newaxis]
    return X, y


def process_nifti_data():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    if not os.path.exists(METADATA):
        print(f"Metadata file missing: {METADATA}")
        return

    df = pd.read_csv(METADATA)

    print("Subjects:", len(df))
    print(df["LABEL"].value_counts())

    for _, row in df.iterrows():

        file_id = str(row["FILE_ID"]).strip()
        label = int(row["LABEL"])

        input_file = os.path.join(
            RAW_DIR,
            f"{file_id}_alff.nii.gz"
        )

        if not os.path.exists(input_file):
            print(f"Missing: {input_file}")
            continue

        img = nib.load(input_file)

        data = img.get_fdata().astype(np.float32)

        data = np.nan_to_num(data)

        mean = data.mean()
        std = data.std()

        if std > 0:
            data = (data - mean) / std

        output_file = os.path.join(
            OUTPUT_DIR,
            f"{file_id}_label{label}.npy"
        )

        np.save(output_file, data)

        print(f"Processed: {file_id} | Label: {label}")

    print("\nPreprocessing completed.")

if __name__ == "__main__":
    process_nifti_data()