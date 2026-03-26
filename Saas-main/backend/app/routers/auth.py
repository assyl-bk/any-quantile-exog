from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError, IntegrityError
from datetime import datetime, timedelta
from typing import cast
from app.database import get_db
from app.models import User, APIKey
from app.schemas import UserCreate, UserLogin, Token, UserResponse, RoleCapabilitiesResponse, RolesMatrixResponse, UserRole
from app.roles import get_role_capabilities, get_all_role_capabilities
from app.security import (
    get_password_hash,
    verify_password,
    create_access_token,
    get_current_active_user
)

router = APIRouter()

@router.post("/signup", response_model=Token, status_code=status.HTTP_201_CREATED)
async def signup(user_data: UserCreate, db: Session = Depends(get_db)):
    """Register a new user and automatically create their first API key"""
    normalized_email = user_data.email.strip().lower()

    # Check if user already exists
    existing_user = db.query(User).filter(User.email == normalized_email).first()
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered"
        )
    
    # Create new user
    hashed_password = get_password_hash(user_data.password)
    new_user = User(
        email=normalized_email,
        name=user_data.name,
        role=user_data.role,
        hashed_password=hashed_password
    )

    try:
        db.add(new_user)
        db.flush()

        # Automatically create a default API key for the new user
        api_key = APIKey.generate_key()
        new_api_key = APIKey(
            user_id=new_user.id,
            name=f"Default API Key for {new_user.name}",
            key=api_key,
            prefix=api_key[:12],  # Store prefix for display (e.g., "efp_xxxxxxxx")
            permissions="read",
            expires_at=datetime.utcnow() + timedelta(days=365)  # Expires in 1 year
        )

        db.add(new_api_key)
        db.commit()
        db.refresh(new_user)
    except IntegrityError:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered"
        )
    except SQLAlchemyError:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Could not create account"
        )
    
    # Create access token
    access_token = create_access_token(data={"sub": new_user.email})
    
    return Token(
        access_token=access_token,
        token_type="bearer",
        user=UserResponse.model_validate(new_user),
        api_key=api_key,
    )

@router.post("/login", response_model=Token)
async def login(credentials: UserLogin, db: Session = Depends(get_db)):
    """Authenticate user and return access token"""
    normalized_email = credentials.email.strip().lower()

    # Find user by email
    try:
        user = db.query(User).filter(User.email == normalized_email).first()
    except SQLAlchemyError:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Could not process login"
        )

    if not user or not verify_password(credentials.password, cast(str, user.hashed_password)):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"}
        )
    
    if not cast(bool, user.is_active):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Account is inactive"
        )
    
    # Create access token
    access_token = create_access_token(data={"sub": user.email})

    # Fetch the user's active API key to include in the response
    active_key = (
        db.query(APIKey)
        .filter(APIKey.user_id == user.id, APIKey.is_active == True)
        .order_by(APIKey.created_at.asc())
        .first()
    )
    api_key_value = cast(str, active_key.key) if active_key else None

    return Token(
        access_token=access_token,
        token_type="bearer",
        user=UserResponse.model_validate(user),
        api_key=api_key_value,
    )

@router.get("/validate", response_model=UserResponse)
async def validate_token(current_user: User = Depends(get_current_active_user)):
    """Validate token and return user info"""
    return UserResponse.model_validate(current_user)

@router.get("/me", response_model=UserResponse)
async def get_current_user_info(current_user: User = Depends(get_current_active_user)):
    """Get current user information"""
    return UserResponse.model_validate(current_user)


@router.get("/capabilities", response_model=RoleCapabilitiesResponse)
async def get_my_capabilities(current_user: User = Depends(get_current_active_user)):
    """Get role-based capabilities for the current authenticated user"""
    role = cast(str, current_user.role)
    return RoleCapabilitiesResponse(
        role=cast(UserRole, role),
        capabilities=get_role_capabilities(role)
    )


@router.get("/roles-matrix", response_model=RolesMatrixResponse)
async def get_roles_matrix(current_user: User = Depends(get_current_active_user)):
    """Return all actor roles and their represented functional needs in the application."""
    return RolesMatrixResponse(
        roles=cast(dict[UserRole, list[str]], get_all_role_capabilities())
    )


@router.get("/my-api-key", response_model=dict)
async def get_my_api_key(
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Return the full API key for the current user (used on session restore)."""
    active_key = (
        db.query(APIKey)
        .filter(APIKey.user_id == current_user.id, APIKey.is_active == True)
        .order_by(APIKey.created_at.asc())
        .first()
    )
    if not active_key:
        return {"api_key": None}
    return {"api_key": cast(str, active_key.key)}