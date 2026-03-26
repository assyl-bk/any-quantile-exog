"""
Initialize the database by creating all tables.
Run this script before starting the API server for the first time.
"""

from app.database import engine, Base
from app.models import User, APIKey

def init_db():
    """Create all tables in the database."""
    print("Creating database tables...")
    Base.metadata.create_all(bind=engine)
    print("Database tables created successfully!")
    print("\nTables created:")
    print("  - users")
    print("  - api_keys")
    print("\nYou can now start the API server with: python main.py")

if __name__ == "__main__":
    init_db()
