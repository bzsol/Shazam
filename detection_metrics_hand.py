import subprocess
import os
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix

def run_identify(database_file, sample_file):
    """
    Run the identify.py script and capture the output.
    """
    try:
        result = subprocess.run(
            ['python', 'identify.py', '-d', database_file, '-i', sample_file],
            capture_output=True,
            text=True
        )
        output = result.stdout.strip()
        print(f"Output from identify.py for {sample_file}: {output}")
        return output
    except Exception as e:
        print(f"Error running identify.py: {e}")
        return "error"  # Returning 'error' to handle exceptions

def calculate_detection_metrics(y_true, y_pred):
    """
    Calculate detection metrics for the identify program.
    """
    if len(set(y_true)) == 1 or len(set(y_pred)) == 1:
        # In case there's only one class in true or predicted labels, add the missing class
        y_true.append(1 - y_true[0])
        y_pred.append(1 - y_pred[0])

    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    accuracy = accuracy_score(y_true, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    specificity = tn / (tn + fp)

    return precision, recall, f1, accuracy, specificity

def main(database_file, samples_dir):
    """
    Main function to process audio files in a directory and calculate detection metrics.
    """
    y_true = []
    y_pred = []

    # Walk through the samples directory and all subdirectories
    for root, _, files in os.walk(samples_dir):
        for file in files:
            if file.endswith('.wav'):
                sample_file = os.path.join(root, file)
                print(f"Processing file: {sample_file}")
                
                # Run identify.py and get the predicted label
                output = run_identify(database_file, sample_file)
                
                # Get user input for the true label based on the output
                while True:
                    user_input = input(f"Is the output '{output}' correct for {file}? (y/n): ").strip().lower()
                    if user_input in ['y', 'n']:
                        true_label = 1 if user_input == 'y' else 0
                        break
                    else:
                        print("Invalid input. Please enter 'y' or 'n'.")
                
                y_true.append(true_label)
                
                # Determine the predicted label based on the output (for simplicity, let's assume '01_Bourgade' indicates a match)
                predicted_label = 1 if '01_Bourgade' in output else 0
                y_pred.append(predicted_label)

    # Calculate detection metrics
    precision, recall, f1, accuracy, specificity = calculate_detection_metrics(y_true, y_pred)

    # Print the results
    print(f'Precision: {precision:.2f}')
    print(f'Recall: {recall:.2f}')
    print(f'F1-Score: {f1:.2f}')
    print(f'Accuracy: {accuracy:.2f}')
    print(f'Specificity: {specificity:.2f}')

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Identify samples in a directory and calculate detection metrics.')
    parser.add_argument('-d', '--database', required=True, help='Database file')
    parser.add_argument('-s', '--samples', required=True, help='Directory containing sample files')
    args = parser.parse_args()
    
    main(args.database, args.samples)
