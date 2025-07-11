import sqlite3
from contextlib import closing
from pathlib import Path

# Define the path for the SQLite database
DATABASE_PATH = Path("database/logs.db")

# Function to initialize the database
def initialize_database():
    with closing(sqlite3.connect(DATABASE_PATH)) as conn:
        cursor = conn.cursor()
        # Create logs table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS emotion_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            emotion TEXT,
            confidence REAL,
            model TEXT,
            session_id TEXT
        )
        ''')
        conn.commit()

# Function to log prediction
def log_prediction(emotion: str, confidence: float, model: str, session_id: str):
    with closing(sqlite3.connect(DATABASE_PATH)) as conn:
        cursor = conn.cursor()
        cursor.execute('''
        INSERT INTO emotion_logs (emotion, confidence, model, session_id)
        VALUES (?, ?, ?, ?)
        ''', (emotion, confidence, model, session_id))
        conn.commit()

# Initialize the database on import
initialize_database()
