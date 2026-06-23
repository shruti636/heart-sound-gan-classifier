import os
import urllib.request
import pandas as pd

def download_dataset_subset(dest_dir='data/raw', num_normal=117, num_abnormal=80):
    os.makedirs(dest_dir, exist_ok=True)
    
    # Official PhysioNet Challenge 2016 REFERENCE file for training-a
    ref_url = "https://physionet.org/files/challenge-2016/1.0.0/training-a/REFERENCE.csv"
    ref_local_path = os.path.join(dest_dir, "REFERENCE_physionet.csv")
    
    print(f"Downloading official REFERENCE labels from: {ref_url}")
    try:
        urllib.request.urlretrieve(ref_url, ref_local_path)
        print("REFERENCE file downloaded successfully.")
    except Exception as e:
        print(f"Error downloading REFERENCE file: {e}")
        return
        
    # Read REFERENCE labels
    # Format is: filename,label (where label is -1 for Normal, 1 for Abnormal)
    df = pd.read_csv(ref_local_path, header=None, names=['filename', 'label'])
    
    # Select normal and abnormal subsets
    # Normal is -1, Abnormal is 1
    normals = df[df['label'] == -1].head(num_normal).copy()
    abnormals = df[df['label'] == 1].head(num_abnormal).copy()
    
    print(f"Selected {len(normals)} normal and {len(abnormals)} abnormal recordings to download.")
    
    selected_recordings = pd.concat([normals, abnormals], ignore_index=True)
    
    # Downloader loop
    base_audio_url = "https://physionet.org/files/challenge-2016/1.0.0/training-a/"
    
    downloaded_records = []
    
    for idx, row in selected_recordings.iterrows():
        rec_name = row['filename']
        label_val = row['label']
        label_str = 'normal' if label_val == -1 else 'abnormal'
        
        wav_filename = f"{rec_name}.wav"
        wav_url = f"{base_audio_url}{wav_filename}"
        wav_dest = os.path.join(dest_dir, wav_filename)
        
        print(f"[{idx+1}/{len(selected_recordings)}] Downloading {wav_filename} ({label_str})...")
        try:
            # Check if already exists to avoid redundant downloads
            if not os.path.exists(wav_dest):
                urllib.request.urlretrieve(wav_url, wav_dest)
            downloaded_records.append({
                'filename': wav_filename,
                'label': label_str
            })
        except Exception as e:
            print(f"Error downloading {wav_filename}: {e}")
            
    # Save local reference.csv in the correct format for preprocessing
    local_ref_df = pd.DataFrame(downloaded_records)
    local_ref_path = os.path.join(dest_dir, "reference.csv")
    local_ref_df.to_csv(local_ref_path, index=False)
    
    print(f"Successfully downloaded {len(local_ref_df)} wav files.")
    print(f"Local reference file written to {local_ref_path}")
    
    # Cleanup intermediate file
    if os.path.exists(ref_local_path):
        os.remove(ref_local_path)

if __name__ == '__main__':
    download_dataset_subset()
