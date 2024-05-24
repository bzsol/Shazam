import os
import time
import numpy as np
import librosa
import scipy.ndimage
import sqlite3
import argparse
import logging

# Configure logging
logging.basicConfig(
    filename='database_builder.log',
    level=logging.DEBUG,  # Set to DEBUG to capture all levels of log messages
    format='%(asctime)s - %(levelname)s - %(module)s - %(funcName)s - %(message)s'
)

def get_spectrogram(y, sr):
    logging.debug("Starting spectrogram generation.")
    try:
        S = np.abs(librosa.stft(y, n_fft=2048, hop_length=512))
        S_db = librosa.amplitude_to_db(S, ref=np.max)
        logging.debug("Spectrogram generation completed.")
        return S_db
    except Exception as e:
        logging.error("Error in get_spectrogram", exc_info=True)
        raise

def find_peaks(spectrogram, threshold=20):
    logging.debug("Starting peak finding.")
    try:
        neighborhood_size = (3, 3)
        data_max = scipy.ndimage.maximum_filter(spectrogram, size=neighborhood_size)
        maxima = (spectrogram == data_max)
        diff = (data_max - spectrogram) > threshold
        maxima[diff] = False
        peaks = np.argwhere(maxima)
        logging.debug("Peak finding completed.")
        return peaks
    except Exception as e:
        logging.error("Error in find_peaks", exc_info=True)
        raise

def generate_hashes_and_insert(peaks, song_id, label, conn, fan_value=15):
    logging.debug(f"Starting hash generation for song_id: {song_id}")
    try:
        cursor = conn.cursor()
        num_peaks = len(peaks)
        for i in range(num_peaks):
            for j in range(1, fan_value):
                if (i + j) < num_peaks:
                    freq1, time1 = peaks[i]
                    freq2, time2 = peaks[i + j]
                    hash_pair = (freq1, freq2, time2 - time1)
                    hash_str = ','.join(map(str, hash_pair))
                    cursor.execute('''
                        INSERT INTO fingerprints (hash, offset, song_id, label)
                        VALUES (?, ?, ?, ?)
                    ''', (hash_str, time1, song_id, label))
        conn.commit()
        logging.info(f"Hashes for song_id {song_id} inserted into database.")
    except Exception as e:
        logging.error("Error in generate_hashes_and_insert", exc_info=True)
        raise

def fingerprint_song_and_insert(file_path, conn):
    logging.info(f"Processing file: {file_path}")
    try:
        y, sr = librosa.load(file_path, sr=None)
        spectrogram = get_spectrogram(y, sr)
        peaks = find_peaks(spectrogram)
        song_id = os.path.splitext(os.path.basename(file_path))[0]
        label = os.path.basename(file_path)
        generate_hashes_and_insert(peaks, song_id, label, conn)
    except Exception as e:
        logging.error(f"Error in fingerprint_song_and_insert for file {file_path}", exc_info=True)
        raise

def setup_database(database_file):
    logging.debug(f"Setting up database: {database_file}")
    try:
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
        conn.close()
        logging.debug("Database setup completed.")
    except Exception as e:
        logging.error("Error in setup_database", exc_info=True)
        raise

def build_database(songs_folder, database_file):
    logging.info(f"Building database from folder: {songs_folder}")
    setup_database(database_file)
    files_to_process = []

    logging.debug(f"Scanning folder: {songs_folder}")

    for root, _, files in os.walk(songs_folder):
        for file in files:
            if file.endswith('.mp3') or file.endswith('.wav'):
                file_path = os.path.join(root, file)
                files_to_process.append(file_path)
                logging.debug(f"Found file: {file_path}")

    if not files_to_process:
        logging.warning("No audio files found in the specified folder.")
        return

    logging.info(f"Starting to process {len(files_to_process)} files.")

    start_time = time.time()
    conn = sqlite3.connect(database_file)
    for file_path in files_to_process:
        try:
            fingerprint_song_and_insert(file_path, conn)
        except Exception as e:
            logging.error(f"Failed to process file: {file_path}", exc_info=True)
    conn.close()
    end_time = time.time()

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

