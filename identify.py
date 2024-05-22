import os
import numpy as np
import librosa
import pickle
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

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
        hashes = set()
        for i in range(len(quantized_mfccs[0])):
            for j in range(1, fan_value):
                if (i + j) < len(quantized_mfccs[0]):
                    hash_pair = tuple(quantized_mfccs[:, i].tolist() + quantized_mfccs[:, i + j].tolist())
                    hashes.add(hash_pair)
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

def load_database(database_file):
    try:
        print("Loading database...")
        with open(database_file, 'rb') as db_file:
            database = pickle.load(db_file)
        print("Database loaded successfully.")
        return database
    except Exception as e:
        print(f"Error loading database: {e}")
        return None

def compare_hashes(sample_hashes, song_hashes):
    return len(sample_hashes.intersection(song_hashes))

def identify_sample(database, sample_file):
    try:
        sample_hashes = fingerprint_song(sample_file)
        
        if not sample_hashes:
            print("Failed to generate hashes for the sample.")
            return None

        matches = {}
        with ThreadPoolExecutor() as executor:
            future_to_song = {executor.submit(compare_hashes, sample_hashes, set(song_hashes)): song_name for song_name, song_hashes in database.items()}
            for future in tqdm(as_completed(future_to_song), total=len(future_to_song), desc='Comparing fingerprints'):
                song_name = future_to_song[future]
                try:
                    match_count = future.result()
                    matches[song_name] = match_count
                except Exception as e:
                    print(f"Error processing {song_name}: {e}")

        best_match = max(matches, key=matches.get)
        return best_match
    except Exception as e:
        print(f"Error identifying sample: {e}")
        return None

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Identify a song sample.')
    parser.add_argument('-d', '--database', required=True, help='Database file')
    parser.add_argument('-i', '--input', required=True, help='Sample file')
    args = parser.parse_args()

    if not os.path.isfile(args.database):
        print(f"The specified database file does not exist: {args.database}")
    else:
        database = load_database(args.database)
        if database:
            if not os.path.isfile(args.input):
                print(f"The specified sample file does not exist: {args.input}")
            else:
                result = identify_sample(database, args.input)
                if result:
                    print(f'Identified song: {result}')
                else:
                    print("Identification failed.")
        else:
            print("Database loading failed.")
