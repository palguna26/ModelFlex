from fastapi import HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from app.firebase_config import verify_firebase_token

security = HTTPBearer()

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Dependency to get current user from Firebase token"""
    token = credentials.credentials
    user = verify_firebase_token(token)
    
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    return user

def require_auth(func):
    """Decorator to require authentication for endpoints"""
    async def wrapper(*args, **kwargs):
        # This will be handled by the Depends(get_current_user) in the endpoint
        return await func(*args, **kwargs)
    return wrapper
