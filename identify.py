import numpy as np
import librosa
import sqlite3
from scipy.ndimage import maximum_filter

def frames_to_samples(frames, sample_rate, hop_length):
    return frames * hop_length / sample_rate

def fft_frequencies(n_fft, sample_rate):
    return np.fft.rfftfreq(n_fft, 1.0 / sample_rate)

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

def generate_hashes_and_insert(cursor, peaks, song_id, label, sample_rate, hop_length, n_fft, fan_value=15):
    try:
        print("Generating hashes and inserting into database...")
        frequencies = fft_frequencies(n_fft, sample_rate)
        num_peaks = len(peaks)
        
        for i in range(num_peaks):
            for j in range(1, fan_value):
                if (i + j) < num_peaks:
                    freq1_index, time1_frame = peaks[i]
                    freq2_index, time2_frame = peaks[i + j]

                    # Convert frame indices and FFT bin indices to physical units
                    freq1 = frequencies[freq1_index]
                    freq2 = frequencies[freq2_index]
                    time1 = frames_to_samples(time1_frame, sample_rate, hop_length)
                    time2 = frames_to_samples(time2_frame, sample_rate, hop_length)

                    time_delta = time2 - time1
                    hash_pair = (freq1, freq2, time_delta)
                    hash_str = ','.join(map(str, hash_pair))

                    cursor.execute('''
                        INSERT INTO fingerprints (hash, offset, song_id, label)
                        VALUES (?, ?, ?, ?)
                    ''', (hash_str, time1, song_id, label))
        
        print("Committing to database...")
        conn.commit()
        print("Hashes inserted and committed successfully.")
    except Exception as e:
        print(f"Error in generate_hashes_and_insert: {e}")
        raise

def fingerprint_song_and_insert(file_path, cursor, song_id, label, sample_rate=44100, hop_length=512, n_fft=2048, fan_value=15):
    try:
        print(f"Fingerprinting song: {file_path}")
        y, sr = librosa.load(file_path, sr=sample_rate)  # Load with specified sampling rate
        spectrogram = get_spectrogram(y, sr)
        peaks = find_peaks(spectrogram)
        generate_hashes_and_insert(cursor, peaks, song_id, label, sr, hop_length, n_fft, fan_value)
    except Exception as e:
        print(f"Error in fingerprint_song_and_insert for file {file_path}: {e}")
        raise

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Fingerprint a song and insert into the database.')
    parser.add_argument('-d', '--database', required=True, help='Database file')
    parser.add_argument('-i', '--input', required=True, help='Input song file')
    parser.add_argument('--song_id', required=True, help='Song ID')
    parser.add_argument('--label', required=True, help='Label for the song')
    args = parser.parse_args()

    if not os.path.isfile(args.database):
        print(f"The specified database file does not exist: {args.database}")
    else:
        if not os.path.isfile(args.input):
            print(f"The specified input file does not exist: {args.input}")
        else:
            conn = sqlite3.connect(args.database)
            cursor = conn.cursor()

            try:
                fingerprint_song_and_insert(args.input, cursor, args.song_id, args.label)
            finally:
                conn.close()
