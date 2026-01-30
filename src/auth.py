"""User authentication and database management."""
import sqlite3
import hashlib
import os
from datetime import datetime
from pathlib import Path

DB_PATH = Path(__file__).parent.parent / 'deepfake.db'

def init_db():
    """Initialize SQLite database with users and uploads tables."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    # Users table
    c.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Uploads table (video uploads and their predictions)
    c.execute('''
        CREATE TABLE IF NOT EXISTS uploads (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            filename TEXT NOT NULL,
            original_filename TEXT NOT NULL,
            upload_path TEXT NOT NULL,
            uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            is_fake_prob REAL,
            is_fake_pred INTEGER,
            confidence REAL,
            num_faces INTEGER,
            detection_details TEXT,
            processed BOOLEAN DEFAULT 0,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    ''')
    
    conn.commit()
    conn.close()

def hash_password(password):
    """Hash password using SHA256."""
    return hashlib.sha256(password.encode()).hexdigest()

def register_user(username, email, password):
    """Register a new user."""
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        password_hash = hash_password(password)
        c.execute('''
            INSERT INTO users (username, email, password_hash)
            VALUES (?, ?, ?)
        ''', (username, email, password_hash))
        conn.commit()
        user_id = c.lastrowid
        conn.close()
        return {'success': True, 'user_id': user_id, 'message': 'User registered successfully'}
    except sqlite3.IntegrityError as e:
        if 'username' in str(e):
            return {'success': False, 'error': 'Username already exists'}
        elif 'email' in str(e):
            return {'success': False, 'error': 'Email already exists'}
        else:
            return {'success': False, 'error': str(e)}
    except Exception as e:
        return {'success': False, 'error': str(e)}

def login_user(username, password):
    """Verify user login credentials."""
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        password_hash = hash_password(password)
        c.execute('SELECT id, username, email FROM users WHERE username = ? AND password_hash = ?',
                  (username, password_hash))
        user = c.fetchone()
        conn.close()
        
        if user:
            return {'success': True, 'user_id': user[0], 'username': user[1], 'email': user[2]}
        else:
            return {'success': False, 'error': 'Invalid username or password'}
    except Exception as e:
        return {'success': False, 'error': str(e)}

def get_user_uploads(user_id):
    """Get all uploads for a user."""
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute('''
            SELECT id, original_filename, uploaded_at, is_fake_pred, is_fake_prob, confidence, num_faces, processed
            FROM uploads
            WHERE user_id = ?
            ORDER BY uploaded_at DESC
        ''', (user_id,))
        uploads = c.fetchall()
        conn.close()
        return [
            {
                'id': u[0],
                'filename': u[1],
                'uploaded_at': u[2],
                'is_fake': u[3],
                'confidence': u[4],
                'prob': u[5],
                'num_faces': u[6],
                'processed': u[7]
            }
            for u in uploads
        ]
    except Exception as e:
        print(f'Error fetching uploads: {e}')
        return []

def save_upload_metadata(user_id, filename, original_filename, upload_path):
    """Save upload metadata to database."""
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute('''
            INSERT INTO uploads (user_id, filename, original_filename, upload_path)
            VALUES (?, ?, ?, ?)
        ''', (user_id, filename, original_filename, upload_path))
        conn.commit()
        upload_id = c.lastrowid
        conn.close()
        return upload_id
    except Exception as e:
        print(f'Error saving upload: {e}')
        return None

def update_upload_results(upload_id, is_fake_prob, is_fake_pred, confidence, num_faces, detection_details):
    """Update upload with model results."""
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute('''
            UPDATE uploads
            SET is_fake_prob = ?, is_fake_pred = ?, confidence = ?, num_faces = ?, detection_details = ?, processed = 1
            WHERE id = ?
        ''', (is_fake_prob, is_fake_pred, confidence, num_faces, detection_details, upload_id))
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        print(f'Error updating upload: {e}')
        return False

def get_upload_details(upload_id):
    """Get detailed information about an upload."""
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute('''
            SELECT id, filename, original_filename, uploaded_at, is_fake_pred, is_fake_prob, confidence, num_faces, detection_details, processed
            FROM uploads
            WHERE id = ?
        ''', (upload_id,))
        result = c.fetchone()
        conn.close()
        
        if result:
            return {
                'id': result[0],
                'filename': result[1],
                'original_filename': result[2],
                'uploaded_at': result[3],
                'is_fake': result[4],
                'prob': result[5],
                'confidence': result[6],
                'num_faces': result[7],
                'detection_details': result[8],
                'processed': result[9]
            }
        return None
    except Exception as e:
        print(f'Error fetching upload details: {e}')
        return None
