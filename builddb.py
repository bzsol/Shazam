import os
import time
import numpy as np
import librosa
import pickle
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import matplotlib.pyplot as plt


def get_mfccs(y, sr):
    try:
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
        mfccs_delta = librosa.feature.delta(mfccs)
        return mfccs_delta
    except Exception as e:
        print(f"Error in get_mfccs: {e}")
        raise

def quantize_mfccs(mfccs, num_bits=8):
    try:
        quantized_mfccs = np.round((mfccs - np.min(mfccs)) / (np.max(mfccs) - np.min(mfccs)) * (2 ** num_bits - 1))
        return quantized_mfccs.astype(np.uint8)
    except Exception as e:
        print(f"Error in quantize_mfccs: {e}")
        raise

def generate_hashes(quantized_mfccs, fan_value=15):
    try:
        hashes = []
        for i in range(len(quantized_mfccs[0])):
            for j in range(1, fan_value):
                if (i + j) < len(quantized_mfccs[0]):
                    hash_pair = tuple(quantized_mfccs[:, i].tolist() + quantized_mfccs[:, i + j].tolist())
                    hashes.append(hash_pair)
        return hashes
    except Exception as e:
        print(f"Error in generate_hashes: {e}")
        raise

def fingerprint_song(file_path):
    try:
        y, sr = librosa.load(file_path, sr=None)  # Load with native sampling rate
        mfccs = get_mfccs(y, sr)
        quantized_mfccs = quantize_mfccs(mfccs)
        hashes = generate_hashes(quantized_mfccs)
        return hashes
    except Exception as e:
        print(f"Error in fingerprint_song for file {file_path}: {e}")
        raise

def process_file(file_path):
    try:
        start_time = time.time()
        hashes = fingerprint_song(file_path)
        end_time = time.time()
        print(f"Successfully processed: {file_path} in {end_time - start_time:.2f} seconds")
        return file_path, hashes
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return file_path, None

def build_database(songs_folder, database_file):
    database = {}
    files_to_process = []

    print(f"Scanning folder: {songs_folder}")

    for root, _, files in os.walk(songs_folder):
        for file in files:
            if file.endswith('.mp3') or file.endswith('.wav'):
                file_path = os.path.join(root, file)
                files_to_process.append(file_path)
                print(f"Found file: {file_path}")

    if not files_to_process:
        print("No audio files found in the specified folder.")
        return

    print(f"Starting to process {len(files_to_process)} files...")

    start_time = time.time()
    try:
        with ThreadPoolExecutor() as executor:
            future_to_file = {executor.submit(process_file, file_path): file_path for file_path in files_to_process}
            for future in as_completed(future_to_file):
                file_path, song_hashes = future.result()
                if song_hashes:
                    database[file_path] = song_hashes
    except Exception as e:
        print(f"Error during processing: {e}")

    end_time = time.time()

    print(f"Saving database to {database_file}")
    try:
        with open(database_file, 'wb') as db_file:
            pickle.dump(database, db_file)
        print("Database built successfully.")
    except Exception as e:
        print(f"Error saving database: {e}")

    print(f"Total time taken: {end_time - start_time:.2f} seconds")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Build fingerprint database.')
    parser.add_argument('-i', '--input', required=True, help='Songs folder')
    parser.add_argument('-o', '--output', required=True, help='Database file')
    args = parser.parse_args()
    
    if not os.path.isdir(args.input):
        print(f"The specified input folder does not exist: {args.input}")
    else:
        print("Starting database build process...")
        build_database(args.input, args.output)
        print("Database build process completed.")
