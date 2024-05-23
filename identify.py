import os
import numpy as np
import librosa
import sqlite3
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

def get_mfccs(y, sr):
    try:
        print("Extracting MFCCs...")
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
        mfccs_delta = librosa.feature.delta(mfccs)
        print("MFCCs extracted successfully.")
        return mfccs_delta
    except Exception as e:
        print(f"Error in get_mfccs: {e}")
        raise

def quantize_mfccs(mfccs, num_bits=8):
    try:
        print("Quantizing MFCCs...")
        quantized_mfccs = np.round((mfccs - np.min(mfccs)) / (np.max(mfccs) - np.min(mfccs)) * (2 ** num_bits - 1))
        print("MFCCs quantized successfully.")
        return quantized_mfccs.astype(np.uint8)
    except Exception as e:
        print(f"Error in quantize_mfccs: {e}")
        raise

def generate_hashes(quantized_mfccs, fan_value=15):
    try:
        print("Generating hashes...")
        hashes = set()
        for i in range(len(quantized_mfccs[0])):
            for j in range(1, fan_value):
                if (i + j) < len(quantized_mfccs[0]):
                    hash_pair = tuple(quantized_mfccs[:, i].tolist() + quantized_mfccs[:, i + j].tolist())
                    hashes.add((hash_pair, i))  # include offset
        print(f"Generated {len(hashes)} hashes.")
        return hashes
    except Exception as e:
        print(f"Error in generate_hashes: {e}")
        raise

def fingerprint_song(file_path):
    try:
        print(f"Fingerprinting song: {file_path}")
        y, sr = librosa.load(file_path, sr=None)  # Load with native sampling rate
        mfccs = get_mfccs(y, sr)
        quantized_mfccs = quantize_mfccs(mfccs)
        hashes = generate_hashes(quantized_mfccs)
        return hashes
    except Exception as e:
        print(f"Error in fingerprint_song for file {file_path}: {e}")
        raise

def insert_sample_hashes(cursor, sample_hashes):
    print("Inserting sample hashes into temporary table...")
    cursor.execute("CREATE TEMP TABLE IF NOT EXISTS sample_hashes (hash TEXT, offset INTEGER)")
    for hash_pair, offset in sample_hashes:
        hash_str = ','.join(map(str, hash_pair))
        cursor.execute("INSERT INTO sample_hashes (hash, offset) VALUES (?, ?)", (hash_str, offset))
    print("Sample hashes inserted.")

def query_database_chunk(database_file, sample_hash_chunk):
    print("Querying database chunk...")
    conn = sqlite3.connect(database_file)
    cursor = conn.cursor()

    insert_sample_hashes(cursor, sample_hash_chunk)
    
    cursor.execute("""
        SELECT f.song_id, f.hash, f.offset, s.offset
        FROM fingerprints f
        JOIN sample_hashes s
        ON f.hash = s.hash
    """)

    results = cursor.fetchall()
    conn.close()
    print(f"Database chunk query returned {len(results)} results.")
    return results

def identify_sample(database_file, sample_file, batch_size=100):
    try:
        print(f"Identifying sample: {sample_file}")
        sample_hashes = fingerprint_song(sample_file)
        
        if not sample_hashes:
            print("Failed to generate hashes for the sample.")
            return None

        sample_hashes = list(sample_hashes)
        results = []

        with ThreadPoolExecutor() as executor:
            futures = []
            for i in range(0, len(sample_hashes), batch_size):
                chunk = sample_hashes[i:i + batch_size]
                futures.append(executor.submit(query_database_chunk, database_file, chunk))
            
            for future in futures:
                chunk_results = future.result()
                print(f"Chunk processed with {len(chunk_results)} results.")
                results.extend(chunk_results)

        if not results:
            print("No matching hashes found in the database.")
            return None

        offset_counter = Counter()
        for song_id, db_hash, db_offset, sample_offset in results:
            offset_diff = db_offset - sample_offset
            offset_counter[(song_id, offset_diff)] += 1

        if not offset_counter:
            print("No similar hashes found in the database.")
            return None

        # Find the song ID with the most common offset difference
        best_match = max(offset_counter.items(), key=lambda x: x[1])
        identified_song_id = best_match[0][0]
        
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
