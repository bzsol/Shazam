import os
import time
import numpy as np
import librosa
import cupy as cp
import cupyx.scipy.ndimage
import sqlite3
import argparse
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed

# Maximum number of concurrent processes
MAX_PROCESSES = 4

# Configure logging
logging.basicConfig(filename='database_builder.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_spectrogram(y, sr):
    try:
        S = np.abs(librosa.stft(y, n_fft=2048, hop_length=512))
        S_db = librosa.amplitude_to_db(S, ref=np.max)
        return S_db
    except Exception as e:
        logging.error(f"Error in get_spectrogram: {e}", exc_info=True)
        raise

def find_peaks(spectrogram, threshold=20):
    try:
        spectrogram_cp = cp.array(spectrogram)
        
        # Identify local maxima in the spectrogram
        neighborhood_size = (3, 3)
        data_max = cupyx.scipy.ndimage.maximum_filter(spectrogram_cp, size=neighborhood_size)
        maxima = (spectrogram_cp == data_max)
        
        # Apply threshold
        diff = (data_max - spectrogram_cp) > threshold
        maxima[diff] = False
        
        peaks = cp.argwhere(maxima)
        peaks = cp.asnumpy(peaks)
        return peaks
    except Exception as e:
        logging.error(f"Error in find_peaks: {e}", exc_info=True)
        raise

def generate_hashes(peaks, fan_value=15):
    try:
        hashes = []
        num_peaks = len(peaks)
        for i in range(num_peaks):
            for j in range(1, fan_value):
                if (i + j) < num_peaks:
                    freq1, time1 = peaks[i]
                    freq2, time2 = peaks[i + j]
                    hash_pair = (freq1, freq2, time2 - time1)
                    hashes.append((hash_pair, time1))  # include offset
        logging.info(f"Total hashes generated: {len(hashes)}")
        return hashes
    except Exception as e:
        logging.error(f"Error in generate_hashes: {e}", exc_info=True)
        raise

def fingerprint_song(file_path):
    try:
        y, sr = librosa.load(file_path, sr=None)  # Load with native sampling rate
        spectrogram = get_spectrogram(y, sr)
        peaks = find_peaks(spectrogram)
        hashes = generate_hashes(peaks)
        return hashes
    except Exception as e:
        logging.error(f"Error in fingerprint_song for file {file_path}: {e}", exc_info=True)
        raise

def process_file(file_path):
    try:
        start_time = time.time()
        hashes = fingerprint_song(file_path)
        end_time = time.time()
        logging.info(f"Successfully processed: {file_path} in {end_time - start_time:.2f} seconds")
        return file_path, hashes
    except Exception as e:
        logging.error(f"Error processing {file_path}: {e}", exc_info=True)
        return file_path, None

def setup_database(database_file):
    conn = sqlite3.connect(database_file)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS fingerprints (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            hash TEXT NOT NULL,
            offset INTEGER NOT NULL,
            song_id TEXT NOT NULL,
            label TEXT NOT NULL
        )
    ''')
    conn.commit()
    return conn

def insert_hashes(conn, song_id, label, hashes):
    cursor = conn.cursor()
    # Use executemany to batch insert for efficiency
    data_to_insert = [
        (','.join(map(str, hash_pair)), offset, song_id, label)
        for hash_pair, offset in hashes
    ]
    cursor.executemany('''
        INSERT INTO fingerprints (hash, offset, song_id, label)
        VALUES (?, ?, ?, ?)
    ''', data_to_insert)
    conn.commit()

def build_database(songs_folder, database_file):
    conn = setup_database(database_file)
    files_to_process = []

    logging.info(f"Scanning folder: {songs_folder}")

    for root, _, files in os.walk(songs_folder):
        for file in files:
            if file.endswith('.mp3') or file.endswith('.wav'):
                file_path = os.path.join(root, file)
                files_to_process.append(file_path)
                logging.info(f"Found file: {file_path}")

    if not files_to_process:
        logging.warning("No audio files found in the specified folder.")
        return

    logging.info(f"Starting to process {len(files_to_process)} files...")

    start_time = time.time()
    try:
        with ProcessPoolExecutor(max_workers=MAX_PROCESSES) as executor:
            future_to_file = {executor.submit(process_file, file_path): file_path for file_path in files_to_process}
            for future in as_completed(future_to_file):
                file_path, song_hashes = future.result()
                if song_hashes:
                    song_id = os.path.splitext(os.path.basename(file_path))[0]
                    label = os.path.basename(file_path)
                    insert_hashes(conn, song_id, label, song_hashes)
    except Exception as e:
        logging.error(f"Error during processing: {e}", exc_info=True)

    end_time = time.time()
    conn.close()

    logging.info(f"Total time taken: {end_time - start_time:.2f} seconds")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Build fingerprint database.')
    parser.add_argument('-i', '--input', required=True, help='Songs folder')
    parser.add_argument('-o', '--output', required=True, help='Database file')
    args = parser.parse_args()
    
    if not os.path.isdir(args.input):
        logging.error(f"The specified input folder does not exist: {args.input}")
    else:
        logging.info("Starting database build process...")
        build_database(args.input, args.output)
        logging.info("Database build process completed.")
