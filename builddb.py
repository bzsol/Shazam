import sqlite3
import os
import logging
import time
import scipy.io.wavfile as wavfile
import argparse
from concurrent.futures import ThreadPoolExecutor
import warnings
import numpy as np
import scipy.fftpack
from scipy.ndimage import maximum_filter
from scipy.ndimage import generate_binary_structure, iterate_structure, binary_erosion

# Ignore warning
warnings.filterwarnings("ignore", category=wavfile.WavFileWarning)

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
        """
        Calculate the spectrogram of audio data.
        """
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
        """
        Find peaks in the spectrogram.
        """
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
        """
        Generate hashes from the peaks.
        """
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



def create_database(database_filename):
    """
    Create a SQLite database with a table for storing fingerprints.

    Parameters:
    - database_filename: The filename for the SQLite database.

    Returns:
    - conn: SQLite connection object.
    - cursor: SQLite cursor object.
    """
    conn = sqlite3.connect(database_filename)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS fingerprints (
            hash TEXT,
            offset REAL,
            song_name TEXT
        )
    ''')
    return conn, cursor

def insert_elements(conn, cursor, values):
    """
    Insert fingerprint elements into the database.

    Parameters:
    - conn: SQLite connection object.
    - cursor: SQLite cursor object.
    - values: List of tuples containing hash, offset, and song name.
    """
    query = "INSERT INTO fingerprints (hash, offset, song_name) VALUES (?, ?, ?)"
    cursor.executemany(query, values)
    conn.commit()

def close_database(conn):
    """
    Save changes and close the database connection.

    Parameters:
    - conn: SQLite connection object.
    """
    conn.close()

def process_song(audio_file):
    """
    Process an audio file and generate fingerprints.

    Parameters:
    - audio_file: Path to the audio file.

    Returns:
    - List of tuples containing hash, offset, and filename.
    """
    filename = os.path.basename(audio_file)[:-5]
    logging.info("Creating fingerprint for: %s", filename)
    _, channels = wavfile.read(audio_file)
    all_hashes = []
    all_offsets = []
    for channel in range(0, 2):
        hashes, offsets = generate_fingerprint(channels[:, channel])
        all_hashes.extend(hashes)
        all_offsets.extend(offsets)
    return [(all_hashes[i], float(all_offsets[i]), filename) for i in range(len(all_hashes))]

def main(args):
    """
    Main function to build the fingerprint database.

    Parameters:
    - args: Command-line arguments containing input folder and output database filename.
    """
    input_folder = args.input_folder
    output_database = args.output_database

    logging.info(f'The input folder is: {input_folder}')
    logging.info(f'Database file name will be: {output_database}')

    try:
        os.remove(output_database)
    except OSError:
        pass

    start_time = time.time()

    conn, cursor = create_database(output_database)

    songs_paths = [os.path.join(input_folder, filename) for filename in os.listdir(input_folder) if filename.endswith('.wav')]

    with ThreadPoolExecutor() as executor:
        results = executor.map(process_song, songs_paths)
        all_song_values = [value for sublist in results for value in sublist]
        insert_elements(conn, cursor, all_song_values)

    close_database(conn)
    end_time = time.time()
    elapsed_time = end_time - start_time
    logging.info("Database creation time was: %.2f seconds", elapsed_time)
    logging.info("All songs have been inserted into the database!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Build fingerprint database.')
    parser.add_argument('-i', '--input_folder', required=True, help='Folder containing songs')
    parser.add_argument('-o', '--output_database', required=True, help='Output database file')
    args = parser.parse_args
