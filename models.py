from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import LargeBinary

# Initialize SQLAlchemy
db = SQLAlchemy()

class User(db.Model):
    """
    User model representing a user in the database.
    Maps to the 'user' table and defines all necessary columns.
    """
    __tablename__ = 'user'  # Table name in the database

    # Columns
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(128), nullable=False)
    account_number = db.Column(db.String(20), unique=True, nullable=False)
    user_type = db.Column(db.String(50), default="Regular User")
    speaker_embedding = db.Column(LargeBinary)  # New field for deep learning embedding
    voice_data = db.Column(LargeBinary)
    training_mfccs = db.Column(LargeBinary)
    dtw_threshold = db.Column(db.Float)
    voice_mean = db.Column(LargeBinary)
    voice_std = db.Column(LargeBinary)

    def set_password(self, password):
        """
        Set the user's password by hashing it with SHA-256.

        Args:
            password (str): The plain-text password to hash.
        """
        from hashlib import sha256
        self.password_hash = sha256(password.encode()).hexdigest()

    def check_password(self, password):
        """
        Check if the provided password matches the stored hash.

        Args:
            password (str): The plain-text password to verify.

        Returns:
            bool: True if the password matches, False otherwise.
        """
        from hashlib import sha256
        return self.password_hash == sha256(password.encode()).hexdigest()

    def __repr__(self):
        """
        String representation of the User object.

        Returns:
            str: A string showing the user's email.
        """
        return f'<User {self.email}>'