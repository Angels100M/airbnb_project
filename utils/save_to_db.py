# בסיס נתונים לשמירת כל השיח בין המשתמש לבוט
import sqlite3
import os

DB_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'chat_history.db')

def create_table():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS prompts_responses (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT,
            user_message TEXT,
            prediction TEXT,
            gpt_response TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()

def save_interaction(user_id, user_message, prediction, gpt_response):
    create_table()  # יוודא שהטבלה קיימת
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        INSERT INTO prompts_responses (user_id, user_message, prediction, gpt_response)
        VALUES (?, ?, ?, ?)
    """, (user_id, user_message, prediction, gpt_response))
    conn.commit()
    conn.close()
