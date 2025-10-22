import firebase_admin
from firebase_admin import credentials, auth, firestore
import os

# Initialize Firebase Admin SDK
# In production, you should use a service account key file
# For now, we'll use the default credentials (requires GOOGLE_APPLICATION_CREDENTIALS env var)
try:
    # Try to initialize with default credentials
    if not firebase_admin._apps:
        cred = credentials.ApplicationDefault()
        firebase_admin.initialize_app(cred)
except Exception as e:
    print(f"Warning: Could not initialize Firebase Admin SDK: {e}")
    print("Please set GOOGLE_APPLICATION_CREDENTIALS environment variable to your service account key file")

# Get Firestore client
db = firestore.client()

def verify_firebase_token(token):
    """Verify Firebase ID token and return user info"""
    try:
        decoded_token = auth.verify_id_token(token)
        return decoded_token
    except Exception as e:
        print(f"Token verification failed: {e}")
        return None

def get_user_uploads(user_id):
    """Get all uploads for a specific user"""
    try:
        uploads_ref = db.collection('model_uploads')
        query = uploads_ref.where('userId', '==', user_id).order_by('createdAt', direction=firestore.Query.DESCENDING)
        docs = query.stream()
        
        uploads = []
        for doc in docs:
            upload_data = doc.to_dict()
            upload_data['id'] = doc.id
            uploads.append(upload_data)
        
        return uploads
    except Exception as e:
        print(f"Error fetching user uploads: {e}")
        return []

def save_upload_metadata(user_id, original_filename, target_device, optimized_filename, metrics):
    """Save upload metadata to Firestore"""
    try:
        upload_data = {
            'userId': user_id,
            'originalFilename': original_filename,
            'targetDevice': target_device,
            'optimizedFilename': optimized_filename,
            'metrics': metrics,
            'createdAt': firestore.SERVER_TIMESTAMP
        }
        
        doc_ref = db.collection('model_uploads').add(upload_data)
        return doc_ref[1].id  # Return the document ID
    except Exception as e:
        print(f"Error saving upload metadata: {e}")
        return None
