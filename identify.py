from collections import namedtuple
import sqlite3
import sys
import logging
import time
import scipy.io.wavfile as wavfile
import argparse
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import scipy.fftpack
from scipy.ndimage import maximum_filter
from scipy.ndimage import generate_binary_structure, iterate_structure, binary_erosion

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def generate_fingerprint(audio_data, fs=44100, frame_size=4096, overlap_ratio=0.5, fan_value=15):
    """
    Generate fingerprints for an audio signal.
    
    Parameters:
    - audio_data: numpy array of audio data.
    - fs: Sampling frequency.
    - frame_size: Size of the FFT window.
    - overlap_ratio: Ratio of overlap between frames.
    - fan_value: Number of peaks to consider for hashing.

    Returns:
    - hashes: List of hash values.
    - offsets: Corresponding list of time offsets.
    """
    def get_spectrogram(audio_data, frame_size, overlap_ratio):
        hop_size = int(frame_size * (1 - overlap_ratio))
        window = np.hanning(frame_size)
        spectrogram = []
        for i in range(0, len(audio_data) - frame_size, hop_size):
            frame = audio_data[i:i + frame_size]
            windowed_frame = frame * window
            spectrum = np.abs(scipy.fftpack.fft(windowed_frame)[:frame_size // 2])
            spectrogram.append(spectrum)
        return np.array(spectrogram).T

    def get_peaks(spectrogram, amp_min=10):
        struct = generate_binary_structure(2, 1)
        neighborhood = iterate_structure(struct, 20)
        local_max = maximum_filter(spectrogram, footprint=neighborhood) == spectrogram
        background = (spectrogram == 0)
        eroded_background = binary_erosion(background, structure=neighborhood, border_value=1)
        detected_peaks = local_max ^ eroded_background
        amps = spectrogram[detected_peaks]
        j, i = np.where(detected_peaks)
        peaks = zip(i, j, amps)
        peaks_filtered = [x for x in peaks if x[2] > amp_min]
        return peaks_filtered

    def generate_hashes(peaks, fan_value=15):
        peaks.sort()
        hashes = []
        offsets = []
        for i in range(len(peaks)):
            for j in range(1, fan_value):
                if (i + j) < len(peaks):
                    freq1 = peaks[i][1]
                    freq2 = peaks[i + j][1]
                    t1 = peaks[i][0]
                    t2 = peaks[i + j][0]
                    time_delta = t2 - t1
                    if 0 <= time_delta <= 200:
                        hash_value = f"{freq1}|{freq2}|{time_delta}"
                        hashes.append(hash_value)
                        offsets.append(t1)
        return hashes, offsets

    spectrogram = get_spectrogram(audio_data, frame_size, overlap_ratio)
    peaks = get_peaks(spectrogram)
    hashes, offsets = generate_hashes(peaks, fan_value)

    return hashes, offsets

HashMatch = namedtuple('HashMatch', ['hash', 'offset', 'song_name'])

def get_input_values():
    """
    Parse command-line arguments to get the database filename and input sample filename.

    Returns:
    - database_filename: Filename of the fingerprint database.
    - input_file: Filename of the input audio sample.
    """
    parser = argparse.ArgumentParser(description='Identify sample against fingerprint database.')
    parser.add_argument('-d', '--database', required=True, help='Database file')
    parser.add_argument('-i', '--input', required=True, help='Input sample file')
    args = parser.parse_args()
    return args.database, args.input

def load_database(database_filename):
    """
    Load the fingerprint database.

    Parameters:
    - database_filename: Filename of the fingerprint database.

    Returns:
    - conn: SQLite connection object.
    - cursor: SQLite cursor object.
    """
    conn = sqlite3.connect(database_filename)
    cursor = conn.cursor()
    return conn, cursor

def search_song(cursor, hashes):
    """
    Search for matching hashes in the fingerprint database.

    Parameters:
    - cursor: SQLite cursor object.
    - hashes: List of hash values to search for.

    Returns:
    - List of HashMatch named tuples represents matching hashes.
    """
    placeholders = ', '.join(['?'] * len(hashes))
    query = f"""
        SELECT hash, offset, song_name
        FROM fingerprints
        WHERE hash IN ({placeholders})
    """
    cursor.execute(query, hashes)
    return [HashMatch(*row) for row in cursor.fetchall()]

def process_audio(audio_file):
    """
    Process an audio file and generate hashes.

    Parameters:
    - audio_file: Path to the audio file.

    Returns:
    - all_hashes: List of hash values generated from the audio.
    - all_offsets: Corresponding list of time offsets for each hash.
    """
    _, channels = wavfile.read(audio_file)
    all_hashes = []
    all_offsets = []
    for channel_index in range(0, 2):
        hashes, offsets = generate_fingerprint(channels[:, channel_index])
        all_hashes.extend(hashes)
        all_offsets.extend(offsets)
    return all_hashes, all_offsets

def find_matches(database_filename, hashes):
    """
    Find matching hashes in the fingerprint database.

    Parameters:
    - database_filename: Filename of the fingerprint database.
    - hashes: List of hash values to search for.

    Returns:
    - matches: List of HashMatch named tuples representing matching hashes.
    """
    conn, cursor = load_database(database_filename)
    matches = search_song(cursor, hashes)
    conn.close()
    return matches

def main():
    """
    Main function to identify a sample from fingerprint database.
    Parameters:
        - args: Command-line arguments containing input database and input song filename to identify.
    """
    database_filename, input_file = get_input_values()
    logging.info(f'Database file: {database_filename}')
    logging.info(f'Input file: {input_file}')

    start_time = time.time()

    try:
        input_hashes, input_offsets = process_audio(input_file)
    except FileNotFoundError:
        logging.error("Input file not found.")
        sys.exit(1)

    logging.info("\nSearching for matches...")
    
    with ThreadPoolExecutor() as executor:
        matches = executor.submit(find_matches, database_filename, input_hashes).result()
    
    hash_dict = {match.hash: match.offset for match in matches}
    relative_offsets = [(match.song_name, match.offset - hash_dict[match.hash]) for match in matches]

    logging.info("Possible hash matches: %d", len(relative_offsets))

    candidate_counts = {}
    max_match_count = 0
    best_candidate_name = ''
    for song_name, offset in relative_offsets:
        candidate_counts.setdefault(song_name, {}).setdefault(offset, 0)
        candidate_counts[song_name][offset] += 1
        if candidate_counts[song_name][offset] > max_match_count:
            max_match_count = candidate_counts[song_name][offset]
            best_candidate_name = song_name

    end_time = time.time()
    elapsed_time = end_time - start_time

    logging.info("\nBest matching song: %s with %d matches", best_candidate_name, max_match_count)
    logging.info("Recognition time: %.2f seconds", elapsed_time)

if __name__ == "__main__":
    main()
