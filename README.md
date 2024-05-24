# Task: Implementation of Shazam – Zsolt Berecz, Martin Bucko

### Introduction:
The implementation of Shazam is a challenging assignment, providing hands-on experience with hashing using spectrograms, databases, and Fourier transformations. It involves multimedia fingerprint creation, essentially for copy detection.

### Design of Utilities:
**Programming Language of Choice:**
For ease of use, availability of libraries, and rapid development, Python was selected as the primary language.

**Parameter Choices:**
- **Window Size of Fourier Transforms (n_fft):** Determines frequency resolution of the spectrogram. A larger window size offers higher frequency resolution but lower time resolution. 2048 is a commonly chosen value.
- **Hop Length:** Number of samples between successive frames in the Short-Time Fourier Transform (STFT). A hop length of 512 samples provides a good trade-off between time resolution and computational efficiency.
- **Neighborhood Size for Peak Detection:** Key factor in determining the extent of the area surrounding each point in the spectrogram analyzed to identify local maxima. A (3, 3) neighborhood is typically used.
- **Amplitude Threshold for Peaks:** Utilized to only consider peaks well above the local noise floor. A threshold of 20 dB is commonly used.
- **Fan Value:** Critical in determining the number of points considered for each peak during hash pair generation. A fan value of 15 indicates that every peak will be paired with the subsequent 15 peaks in the spectrogram.

### Process Description:
**Spectrogram Generation:**
Function: `get_spectrogram(y, sr)`
Parameters: `n_fft=2048`, `hop_length=512`
Purpose: Converts audio signal to STFT spectrogram for detailed efficiency.

**Peak Detection:**
Function: `find_peaks(spectrogram, threshold=20, neighborhood_size=(3, 3))`
Identify significant peaks by comparing points to local neighborhood and applying amplitude threshold to filter out noise.

**Hash Generation:**
Function: `generate_hashes(peaks, fan_value=15)`
Parameters: `fan_value=15`
Purpose: Creates unique hash pairs from detected peaks by pairing each peak with the next `fan_value` peaks to encode frequency and time differences for robust fingerprints.

### Database Schema:
The fingerprinting system uses a database schema with a single table called `fingerprints`.

#### Columns Explanation:
- `id`: INTEGER, PRIMARY KEY AUTOINCREMENT. Uniquely identifies each row.
- `hash`: TEXT, NOT NULL. Stores hash value for frequency peaks pair, essential for audio fingerprint matching.
- `offset`: INTEGER, NOT NULL. Records time offset of first peak in hash pair for accurate matching.
- `song_id`: TEXT, NOT NULL. Unique song identifier derived from file name without extension.
- `label`: TEXT, NOT NULL. Stores song label for debugging or display purposes, file name with extension.

### Detection Metrics:
- Precision
- Recall
- F1-Score
- Accuracy
- Specificity

---

## User Manual

### System Requirements
**Recommended Computer Specs:**
- Dual-core processor
- 8 GB RAM
- 20+ GB storage
- Python 3.6 or newer

**Operating System:** 
- Windows 10, macOS 10.13 (High Sierra), or a modern Linux distribution (e.g., Ubuntu 18.04)

### Installation
1. **Install Required External Libraries:**
   - Create a `requirements.txt` file with the specified content:
     ```
     numpy
     librosa
     scipy
     ```
   - Install the libraries using the following command:
     ```
     pip install -r requirements.txt
     ```

### Using the Program in Command Line Interface:
#### Building the Fingerprint Database:
```
builddb -i songs-folder -o database-file
```
- `-i songs-folder`: Specifies the path to the Song Database folder or the files you want to add to the database.
- `-o database.db`: Specifies the path to the SQLite database file where you want to save it.

#### Identifying a Sample:
```
identify -d database-file -i sample.wav
```
- `-d database.db`: Specifies the path to the SQLite database file.
- `-i sample.wav`: Specifies the path to the audio sample file.

--- 

This user manual provides comprehensive instructions for setting up and using the audio fingerprinting program effectively.
