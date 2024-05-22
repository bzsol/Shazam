import os
import numpy as np
import librosa
import sqlite3
import argparse
from collections import Counter

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
                    hashes.add((hash_pair, i))  # include offset
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

def insert_sample_hashes(cursor, sample_hashes):
    cursor.execute("CREATE TEMP TABLE sample_hashes (hash TEXT, offset INTEGER)")
    for hash_pair, offset in sample_hashes:
        hash_str = ','.join(map(str, hash_pair))
        cursor.execute("INSERT INTO sample_hashes (hash, offset) VALUES (?, ?)", (hash_str, offset))

def calculate_similarity(hash1, hash2):
    match_count = sum(1 for a, b in zip(hash1, hash2) if a == b)
    return match_count / len(hash1)

def identify_sample(database_file, sample_file):
    try:
        sample_hashes = fingerprint_song(sample_file)
        
        if not sample_hashes:
            print("Failed to generate hashes for the sample.")
            return None

        conn = sqlite3.connect(database_file)
        cursor = conn.cursor()
        
        # Insert sample hashes into a temporary table
        print("Inserting sample hashes into temporary table...")
        insert_sample_hashes(cursor, sample_hashes)
        
        # Perform SQL query to find matches
        print("Querying database for matches...")
        cursor.execute("""
            SELECT f.song_id, f.hash, s.hash, s.offset - f.offset as offset_diff
            FROM fingerprints f
            JOIN sample_hashes s
        """)
        
        results = cursor.fetchall()
        conn.close()

        if not results:
            print("No matching hashes found in the database.")
            return None

        threshold = 0.8
        offset_counter = Counter()
        
        for song_id, db_hash, sample_hash, offset_diff in results:
            db_hash_list = list(map(int, db_hash.split(',')))
            sample_hash_list = list(map(int, sample_hash.split(',')))
            
            if calculate_similarity(db_hash_list, sample_hash_list) >= threshold:
                offset_counter[(song_id, offset_diff)] += 1

        if not offset_counter:
            print("No similar hashes found in the database.")
            return None

        # Find the most common (song_id, offset_diff)
        most_common = offset_counter.most_common(1)[0][0]
        identified_song_id = most_common[0]
        
        return identified_song_id

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
        if not os.path.isfile(args.input):
            print(f"The specified sample file does not exist: {args.input}")
        else:
            result = identify_sample(args.database, args.input)
            if result:
                print(f'Identified song: {result}')
            else:
                print("Identification failed.")
