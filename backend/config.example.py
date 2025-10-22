# Firebase Configuration Example
# Copy this file to config.py and update with your actual values

import os

# Firebase Configuration
GOOGLE_APPLICATION_CREDENTIALS = os.getenv('GOOGLE_APPLICATION_CREDENTIALS', './firebase-service-account.json')
FIREBASE_PROJECT_ID = os.getenv('FIREBASE_PROJECT_ID', 'your-project-id')

# File storage directories
UPLOAD_DIR = os.getenv('UPLOAD_DIR', './uploads')
OPTIMIZED_DIR = os.getenv('OPTIMIZED_DIR', './optimized')

# CORS settings
ALLOWED_ORIGINS = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:3000",
    "http://127.0.0.1:3000"
]
