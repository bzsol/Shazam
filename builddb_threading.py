import os
import time
import numpy as np
import librosa
import scipy.ndimage
import sqlite3
import argparse
import logging
from concurrent.futures import ThreadPoolExecutor
from threading import Lock

# Configure logging
logging.basicConfig(
    filename='database_builder.log',
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(module)s - %(funcName)s - %(message)s'
)

db_lock = Lock()

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

def generate_hashes_and_insert(peaks, song_id, label, db_file, fan_value=15):
    logging.debug(f"Starting hash generation for song_id: {song_id}")
    try:
        with db_lock:
            conn = sqlite3.connect(db_file)
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
            conn.close()
            logging.info(f"Hashes for song_id {song_id} inserted into database.")
    except Exception as e:
        logging.error("Error in generate_hashes_and_insert", exc_info=True)
        raise

def fingerprint_song_and_insert(file_path, db_file):
    logging.info(f"Processing file: {file_path}")
    try:
        y, sr = librosa.load(file_path, sr=None)
        spectrogram = get_spectrogram(y, sr)
        peaks = find_peaks(spectrogram)
        song_id = os.path.splitext(os.path.basename(file_path))[0]
        label = os.path.basename(file_path)
        generate_hashes_and_insert(peaks, song_id, label, db_file)
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
    
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(fingerprint_song_and_insert, file_path, database_file) for file_path in files_to_process]
        for future in futures:
            try:
                future.result()
            except Exception as e:
                logging.error(f"Error in processing a file with threading: {e}")

    end_time = time.time()

    logging.info(f"Total time taken: {end_time - start_time:.2f} seconds")

def insert_sample_hashes(cursor, sample_hashes):
    logging.debug("Inserting sample hashes into temporary table.")
    cursor.execute("CREATE TEMP TABLE IF NOT EXISTS sample_hashes (hash TEXT, offset INTEGER)")
    for hash_pair, offset in sample_hashes:
        hash_str = ','.join(map(str, hash_pair))
        cursor.execute("INSERT INTO sample_hashes (hash, offset) VALUES (?, ?)", (hash_str, offset))
    logging.debug("Sample hashes inserted.")

def query_database_chunk(database_file, sample_hash_chunk):
    logging.debug("Querying database chunk.")
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
    logging.debug(f"Database chunk query returned {len(results)} results.")
    return results

def identify_sample(database_file, sample_file, batch_size=100):
    try:
        logging.info(f"Identifying sample: {sample_file}")
        sample_hashes = fingerprint_song_and_insert(sample_file, database_file)
        
        if not sample_hashes:
            logging.warning("Failed to generate hashes for the sample.")
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
                logging.debug(f"Chunk processed with {len(chunk_results)} results.")
                results.extend(chunk_results)

        if not results:
            logging.warning("No matching hashes found in the database.")
            return None

        offset_counter = Counter()
        for song_id, db_hash, db_offset, sample_offset in results:
            db_offset = int(db_offset)
            sample_offset = int(sample_offset)
            offset_diff = db_offset - sample_offset
            offset_counter[(song_id, offset_diff)] += 1

        if not offset_counter:
            logging.warning("No similar hashes found in the database.")
            return None

        best_match = max(offset_counter.items(), key=lambda x: x[1])
        identified_song_id = best_match[0][0]
        
        return identified_song_id

    except Exception as e:
        logging.error(f"Error identifying sample: {e}", exc_info=True)
        return None

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Build fingerprint database and identify samples.')
    parser.add_argument('-i', '--input', required=True, help='Songs folder or sample file')
    parser.add_argument('-o', '--output', required=True, help='Database file')
    parser.add_argument('--identify', action='store_true', help='Identify sample')
    args = parser.parse_args()

    if not os.path.isdir(args.input) and not os.path.isfile(args.input):
        logging.error(f"The specified input path does not exist: {args.input}")
    else:
        if args.identify:
            if os.path.isfile(args.input):
                result = identify_sample(args.output, args.input)
                if result:
                    logging.info(f"Identified song ID: {result}")
                else:
                    logging.info("Identification was not precise. No matching song found.")
            else:
                logging.error("For identification, the input should be a single sample file.")
        else:
            if os.path.isdir(args.input):
                logging.info("Starting database build process...")
                build_database(args.input, args.output)
                logging.info("Database build process completed.")
            else:
                logging.error("For database building, the input should be a folder of songs.")
