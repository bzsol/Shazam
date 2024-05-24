import os
import time
import numpy as np
import librosa
import scipy.ndimage
import sqlite3
import argparse
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from sqlite3 import OperationalError

# Configure logging
logging.basicConfig(filename='database_builder.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_spectrogram(y, sr):
    try:
        S = np.abs(librosa.stft(y, n_fft=2048, hop_length=512))
        S_db = librosa.amplitude_to_db(S, ref=np.max)
        logging.debug(f"Spectrogram calculated with shape {S_db.shape}")
        return S_db
    except Exception as e:
        logging.error(f"Error in get_spectrogram: {e}", exc_info=True)
        raise

def find_peaks(spectrogram, threshold=20):
    try:
        neighborhood_size = (3, 3)
        data_max = scipy.ndimage.maximum_filter(spectrogram, size=neighborhood_size)
        maxima = (spectrogram == data_max)
        diff = (data_max - spectrogram) > threshold
        maxima[diff] = False
        peaks = np.argwhere(maxima)
        logging.debug(f"Peaks found: {len(peaks)}")
        return peaks
    except Exception as e:
        logging.error(f"Error in find_peaks: {e}", exc_info=True)
        raise

def generate_hashes_and_insert(peaks, song_id, label, conn, fan_value=15, max_retries=5):
    retries = 0
    while retries < max_retries:
        try:
            cursor = conn.cursor()
            num_peaks = len(peaks)
            for i in range(num_peaks):
                for j in range(1, fan_value):
                    if (i + j) < num_peaks:
                        freq1_index, time1_frame = peaks[i]
                        freq2_index, time2_frame = peaks[i + j]

                        # Convert frame indices and FFT bin indices to physical units
                        freq1 = librosa.fft_frequencies(freq1_index, 2048)
                        freq2 = librosa.fft_frequencies(freq2_index, 2048)
                        time1 = librosa.frames_to_samples(time1_frame, 2048, 512)
                        time2 = librosa.frames_to_samples(time2_frame, 2048, 512)

                        hash_pair = (freq1, freq2, time2 - time1)
                        hash_str = ','.join(map(str, hash_pair))
                        cursor.execute('''
                            INSERT INTO fingerprints (hash, offset, song_id, label)
                            VALUES (?, ?, ?, ?)
                        ''', (hash_str, time1, song_id, label))
            conn.commit()
            logging.info(f"Hashes for song_id {song_id} inserted into database.")
            return
        except OperationalError as e:
            if 'database is locked' in str(e):
                retries += 0.5
                sleep_time = 2 ** retries  # Exponential backoff
                logging.warning(f"Database is locked, retrying in {sleep_time} seconds...")
                time.sleep(sleep_time)
            else:
                logging.error(f"OperationalError in generate_hashes_and_insert: {e}", exc_info=True)
                raise
        except Exception as e:
            logging.error(f"Error in generate_hashes_and_insert: {e}", exc_info=True)
            raise


def fingerprint_song_and_insert(file_path, database_file):
    try:
        y, sr = librosa.load(file_path, sr=None)
        spectrogram = get_spectrogram(y, sr)
        peaks = find_peaks(spectrogram)
        song_id = os.path.splitext(os.path.basename(file_path))[0]
        label = os.path.basename(file_path)
        
        conn = sqlite3.connect(database_file)  # Create a new connection for each thread
        try:
            generate_hashes_and_insert(peaks, song_id, label, conn)
        finally:
            conn.close()  # Close the connection after use
    except Exception as e:
        logging.error(f"Error in fingerprint_song_and_insert for file {file_path}: {e}", exc_info=True)
        raise

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
    conn.close()
    logging.info(f"Database setup completed for {database_file}")

def build_database(songs_folder, database_file, num_threads=4):
    setup_database(database_file)
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
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            future_to_file = {executor.submit(fingerprint_song_and_insert, file_path, database_file): file_path for file_path in files_to_process}
            for future in as_completed(future_to_file):
                file_path = future_to_file[future]
                try:
                    future.result()
                    logging.info(f"Successfully processed {file_path}")
                except Exception as e:
                    logging.error(f"Error processing {file_path}: {e}", exc_info=True)
    except Exception as e:
        logging.error(f"Error during processing: {e}", exc_info=True)

    end_time = time.time()
    logging.info(f"Total time taken: {end_time - start_time:.2f} seconds")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Build fingerprint database.')
    parser.add_argument('-i', '--input', required=True, help='Songs folder')
    parser.add_argument('-o', '--output', required=True, help='Database file')
    parser.add_argument('-t', '--threads', type=int, default=4, help='Number of threads for parallel processing')
    args = parser.parse_args()
    
    if not os.path.isdir(args.input):
        logging.error(f"The specified input folder does not exist: {args.input}")
    else:
        logging.info("Starting database build process...")
        build_database(args.input, args.output, args.threads)
        logging.info("Database build process completed.")
