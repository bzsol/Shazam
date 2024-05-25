import os
import numpy as np
import librosa
import sqlite3
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from scipy.ndimage import maximum_filter

def get_spectrogram(y, sr):
    try:
        print("Extracting spectrogram...")
        S = np.abs(librosa.stft(y))
        print("Spectrogram extracted successfully.")
        return S
    except Exception as e:
        print(f"Error in get_spectrogram: {e}")
        raise

def find_peaks(spectrogram, threshold=0.8):
    try:
        print("Finding peaks in spectrogram...")
        local_max = maximum_filter(spectrogram, size=20) == spectrogram
        background = (spectrogram == 0)
        eroded_background = maximum_filter(background, size=20) == background
        detected_peaks = local_max ^ eroded_background
        
        peaks = np.argwhere(detected_peaks)
        peaks = peaks[spectrogram[detected_peaks] > np.max(spectrogram) * threshold]
        print(f"Found {len(peaks)} peaks.")
        return peaks
    except Exception as e:
        print(f"Error in find_peaks: {e}")
        raise

def generate_hashes(peaks, sr, fan_value=15):
    try:
        print("Generating hashes...")
        hashes = []
        num_peaks = len(peaks)
        sample_rate = sr
        hop_length = 512
        n_fft = 2048
        
        # Generate FFT frequencies
        freqs = librosa.fft_frequencies(sr=sample_rate, n_fft=n_fft)
        
        # Generate a list of time samples for all frames
        max_time_frame = max(peaks, key=lambda x: x[1])[1]
        times = librosa.frames_to_samples(range(max_time_frame + 1), hop_length=hop_length)
        
        for i in range(num_peaks):
            for j in range(1, fan_value):
                if (i + j) < num_peaks:
                    freq1_index, time1_frame = peaks[i]
                    freq2_index, time2_frame = peaks[i + j]

                    freq1 = freqs[freq1_index]
                    freq2 = freqs[freq2_index]
                    time1 = times[time1_frame] / sample_rate
                    time2 = times[time2_frame] / sample_rate
                    
                    hash_pair = (freq1, freq2, time2 - time1)
                    hashes.append((hash_pair, time1))
        
        print(f"Total hashes generated: {len(hashes)}")
        return hashes
    except Exception as e:
        print(f"Error in generate_hashes: {e}")
        raise



def fingerprint_song(file_path):
    try:
        print(f"Fingerprinting song: {file_path}")
        y, sr = librosa.load(file_path, sr=None)  # Load with native sampling rate
        spectrogram = get_spectrogram(y, sr)
        peaks = find_peaks(spectrogram)
        hashes = generate_hashes(peaks,sr)
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
            # Ensure db_offset and sample_offset are decoded to integers
            db_offset = int.from_bytes(db_offset, byteorder='big', signed=True) if isinstance(db_offset, bytes) else int(db_offset)
            sample_offset = int.from_bytes(sample_offset, byteorder='big', signed=True) if isinstance(sample_offset, bytes) else int(sample_offset)
            offset_diff = db_offset - sample_offset
            offset_counter[(song_id, offset_diff)] += 1

        if not offset_counter:
            print("No similar hashes found in the database.")
            return None

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
