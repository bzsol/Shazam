import sqlite3

def check_database_integrity(database_file):
    try:
        conn = sqlite3.connect(database_file)
        cursor = conn.cursor()

        # Query the fingerprints table
        cursor.execute("SELECT * FROM fingerprints")
        rows = cursor.fetchall()

        if not rows:
            print("No data found in the fingerprints table.")
        else:
            print("Contents of the fingerprints table:")
            for row in rows:
                print(row)

        conn.close()

    except Exception as e:
        print(f"Error checking database integrity: {e}")

if __name__ == "__main__":
    database_file = "db.sql"  # Replace with the actual database file name
    check_database_integrity(database_file)