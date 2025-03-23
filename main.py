import os
import base64
import tempfile
from datetime import datetime
import re
import logging
import pickle
import random

from flask import Flask, request, render_template, session, redirect, url_for, flash, Response
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from dotenv import load_dotenv
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import smtplib

import librosa
import numpy as np
from speechbrain.inference import EncoderClassifier
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
import noisereduce as nr
import torch
from sklearn.preprocessing import normalize

# Load environment variables from email.env
load_dotenv('email.env')

# Initialize Flask app
app = Flask(__name__, template_folder='templates')
app.secret_key = os.getenv("FLASK_SECRET_KEY", "default_secret_key")

# Database configuration
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///site.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Initialize database and migration
db = SQLAlchemy(app)
migrate = Migrate(app, db)

# User model
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(128), nullable=False)
    account_number = db.Column(db.String(20), unique=True, nullable=False)
    user_type = db.Column(db.String(50), default="Regular User")
    speaker_embedding = db.Column(db.LargeBinary)  # Speaker embedding from ECAPA-TDNN
    training_mfccs = db.Column(db.LargeBinary)    # MFCC features for DTW
    dtw_threshold = db.Column(db.Float)           # Threshold for phrase similarity
    # Optional fields from the second document (uncomment if needed)
    # voice_mean = db.Column(db.LargeBinary)
    # voice_std = db.Column(db.LargeBinary)

    def set_password(self, password):
        from hashlib import sha256
        self.password_hash = sha256(password.encode()).hexdigest()

    def check_password(self, password):
        from hashlib import sha256
        return self.password_hash == sha256(password.encode()).hexdigest()

# AudioRecording model with cascade delete
class AudioRecording(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    raw_audio = db.Column(db.LargeBinary, nullable=False)
    user = db.relationship('User', backref=db.backref('audio_recordings', lazy=True, cascade="all, delete-orphan"))

# Create database tables
with app.app_context():
    db.create_all()

# Logging setup
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Email configuration
SENDER_EMAIL = os.getenv("SENDER_EMAIL")
SENDER_PASSWORD = os.getenv("SENDER_PASSWORD")

# Load pre-trained speaker recognition model
classifier = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb")

def send_email_otp(email, otp):
    """Send an OTP to the user's email."""
    message = MIMEMultipart("alternative")
    message["Subject"] = "Your OTP Code for 3FA"
    message["From"] = SENDER_EMAIL
    message["To"] = email
    text = f"Your OTP code is: {otp}\nPlease use this code to complete your authentication."
    part = MIMEText(text, "plain")
    message.attach(part)
    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(SENDER_EMAIL, SENDER_PASSWORD)
            server.sendmail(SENDER_EMAIL, email, message.as_string())
        logger.debug(f"OTP sent successfully to {email}")
        return True
    except Exception as e:
        logger.error(f"Error sending OTP: {e}")
        return False

def load_audio(audio_data):
    """Load audio from bytes at 16kHz sampling rate."""
    try:
        with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as tmp:
            tmp.write(audio_data)
            tmp_filename = tmp.name
        try:
            y, sr = librosa.load(tmp_filename, sr=16000)
            logger.debug(f"Loaded audio from temporary file. Audio shape: {y.shape}, sr={sr}")
            return y
        finally:
            os.remove(tmp_filename)
    except Exception as e:
        logger.error(f"Error loading audio: {e}")
        raise e

def extract_features(audio_data):
    """Extract MFCC features and waveform from audio bytes."""
    y = load_audio(audio_data)
    y = nr.reduce_noise(y=y, sr=16000)
    y = y / np.max(np.abs(y))
    y, _ = librosa.effects.trim(y, top_db=20)
    if len(y) < 16000 * 3:
        logger.warning(f"Trimmed audio too short: {len(y)} samples. Consider re-recording.")
    min_samples = 16000 * 5
    if len(y) < min_samples:
        padding = np.zeros(min_samples - len(y))
        y = np.concatenate([y, padding])
        logger.debug(f"Padded audio from {len(y)-len(padding)} to {min_samples} samples")
    mfcc = librosa.feature.mfcc(y=y, sr=16000, n_mfcc=40, hop_length=50, n_fft=400)
    delta_mfcc = librosa.feature.delta(mfcc)
    delta2_mfcc = librosa.feature.delta(mfcc, order=2)
    features = np.hstack([mfcc.T, delta_mfcc.T, delta2_mfcc.T])
    logger.debug(f"Extracted features: shape={features.shape}")
    if features.shape[0] < 100:
        raise ValueError("Insufficient audio data for feature extraction. Record a longer phrase.")
    return features, y

def train_voice_template(email, all_audio_data):
    """Train and store the user's voice template using multiple audio samples."""
    user = User.query.filter_by(email=email).first()
    if not user:
        raise ValueError(f"User not found: {email}")

    all_features = []
    all_embeddings = []
    for audio_data in all_audio_data:
        features, y = extract_features(audio_data)
        all_features.append(features)
        waveform = torch.from_numpy(y).unsqueeze(0).float()
        embedding = classifier.encode_batch(waveform).squeeze().cpu().numpy()
        all_embeddings.append(embedding)
    mean_embedding = np.mean(all_embeddings, axis=0)
    mean_embedding = normalize(mean_embedding.reshape(1, -1)).flatten()
    user.speaker_embedding = pickle.dumps(mean_embedding)
    max_avg_dtw = 0.0
    for i in range(len(all_features)):
        for j in range(i + 1, len(all_features)):
            distance, path = fastdtw(all_features[i], all_features[j], dist=euclidean)
            avg_dtw = distance / len(path)
            if avg_dtw > max_avg_dtw:
                max_avg_dtw = avg_dtw
    user.dtw_threshold = max_avg_dtw * 1.5
    user.training_mfccs = pickle.dumps(all_features)
    # Optional: Add voice_mean and voice_std (uncomment if fields are added to User model)
    # all_mfccs = [extract_features(audio)[0] for audio in all_audio_data]
    # user.voice_mean = pickle.dumps(np.mean(all_mfccs, axis=0))
    # user.voice_std = pickle.dumps(np.std(all_mfccs, axis=0))
    db.session.commit()
    logger.debug(f"Voice template and data saved for {email}")

def verify_voice(email, audio_data):
    """Verify the user's voice and phrase using speaker embedding and DTW."""
    user = User.query.filter_by(email=email).first()
    if not user or not user.speaker_embedding or not user.training_mfccs or user.dtw_threshold is None:
        logger.debug(f"User not found or incomplete voice data for {email}")
        return False

    try:
        stored_embedding = pickle.loads(user.speaker_embedding)
        training_mfccs = pickle.loads(user.training_mfccs)
    except Exception as e:
        logger.error(f"Error loading voice data for {email}: {e}")
        return False

    features, y = extract_features(audio_data)
    waveform = torch.from_numpy(y).unsqueeze(0).float()
    new_embedding = classifier.encode_batch(waveform).squeeze().cpu().numpy()
    new_embedding = normalize(new_embedding.reshape(1, -1)).flatten()
    similarity = np.dot(new_embedding, stored_embedding)
    logger.debug(f"Similarity score for {email}: {similarity}")
    min_avg_dtw = min([fastdtw(features, train_mfcc, dist=euclidean)[0] / len(features) for train_mfcc in training_mfccs])
    logger.debug(f"Min avg DTW for {email}: {min_avg_dtw}, threshold: {user.dtw_threshold}")
    other_users = User.query.filter(User.id != user.id).all()
    other_embeddings = [pickle.loads(u.speaker_embedding) for u in other_users if u.speaker_embedding]
    max_other_similarity = max([np.dot(new_embedding, other_emb) for other_emb in other_embeddings]) if other_embeddings else -1
    if similarity > max_other_similarity + 0.1 and similarity > 0.8 and min_avg_dtw < user.dtw_threshold:
        logger.info(f"Voice and phrase verified for {email}")
        return True
    else:
        logger.info(f"Voice rejected for {email}")
        return False

def login_time(email):
    """Log the user's login time to a CSV file."""
    with open('Users_login_time.csv', 'a') as f:
        f.write(f"{email},{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

# Routes
@app.route('/')
def index():
    return redirect(url_for('login'))

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        email = request.form['email'].strip().lower()
        password = request.form['password']
        logger.debug(f"Registering user: {email}")

        upper_case_regex = re.compile(r'[A-Z]')
        special_char_regex = re.compile(r'[!@#$%^&*(),.?":{}|<>]')

        if len(password) != 10:
            flash("Password must be exactly 10 characters long", "danger")
            return render_template('register.html', error="Password must be exactly 10 characters long")
        if not upper_case_regex.search(password):
            flash("Password must contain at least one uppercase letter", "danger")
            return render_template('register.html', error="Password must contain at least one uppercase letter")
        if not special_char_regex.search(password):
            flash("Password must contain at least one special character", "danger")
            return render_template('register.html', error="Password must contain at least one special character")

        if User.query.filter_by(email=email).first():
            flash("Email already exists", "danger")
            return render_template('register.html', error="Email already exists")

        while True:
            account_number = f"#{random.randint(1000000000, 9999999999)}"
            if not User.query.filter_by(account_number=account_number).first():
                break

        new_user = User(email=email, account_number=account_number, user_type="Regular User")
        new_user.set_password(password)
        db.session.add(new_user)
        db.session.commit()
        logger.debug(f"User registered: {email}")
        return redirect(url_for('train_voice', email=email))
    return render_template('register.html')

@app.route('/train_voice', methods=['GET', 'POST'])
def train_voice():
    if request.method == 'GET':
        email = request.args.get('email')
        if not email:
            return "Email not provided", 400
        session['train_step'] = 1
        session['audio_paths'] = []
        return render_template('train_voice.html', email=email, step=1, total_steps=10)
    elif request.method == 'POST':
        email = request.form.get('email')
        if not email:
            return "Email not provided", 400
        audio_data = request.form.get('audio_data')
        if audio_data:
            try:
                if not audio_data.startswith('data:audio/'):
                    raise ValueError("Invalid audio data format.")
                base64_string = audio_data.split(',')[1]
                audio_bytes = base64.b64decode(base64_string)
                with tempfile.NamedTemporaryFile(delete=False, suffix='.webm') as tmp:
                    tmp.write(audio_bytes)
                    session['audio_paths'].append(tmp.name)
                session['train_step'] = session.get('train_step', 1) + 1
                if session['train_step'] > 10:
                    all_audio_data = [open(path, 'rb').read() for path in session['audio_paths']]
                    train_voice_template(email, all_audio_data)
                    user = User.query.filter_by(email=email).first()
                    for audio_bytes in all_audio_data:
                        recording = AudioRecording(user_id=user.id, raw_audio=audio_bytes)
                        db.session.add(recording)
                    db.session.commit()
                    for path in session['audio_paths']:
                        os.remove(path)
                    session.pop('train_step')
                    session.pop('audio_paths')
                    return redirect(url_for('login'))
                return render_template('train_voice.html', email=email, step=session['train_step'], total_steps=10)
            except Exception as e:
                logger.error(f"Error training voice for {email}: {e}")
                return render_template('train_voice.html', email=email, error=str(e))
        return render_template('train_voice.html', email=email, error="No audio data received")

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email'].strip().lower()
        password = request.form['password']
        user = User.query.filter_by(email=email).first()
        if user and user.check_password(password):
            otp = str(random.randint(100000, 999999))
            session['otp'] = otp
            session['email'] = email
            if send_email_otp(email, otp):
                return render_template('second.html', email=email)
            return render_template('login.html', error="Failed to send OTP")
        return render_template('login.html', error="Invalid credentials")
    return render_template('login.html')

@app.route('/verify_otp', methods=['POST'])
def verify_otp():
    email = request.form['email']
    user_otp = request.form['otp']
    if session.get('otp') == user_otp:
        session.pop('otp')
        return render_template('third.html', email=email)
    return render_template('third.html', email=email, error="Invalid OTP")

@app.route('/verify_voice', methods=['POST'])
def verify_voice_route():
    email = request.form.get('email')
    audio_data = request.form.get('audio_data')
    if not audio_data:
        return render_template('third.html', email=email, error="No audio data received")
    try:
        if not audio_data.startswith('data:audio/'):
            raise ValueError("Invalid audio data format.")
        base64_string = audio_data.split(',')[1]
        audio_bytes = base64.b64decode(base64_string)
    except Exception as e:
        logger.error(f"Error decoding audio for {email}: {e}")
        return render_template('third.html', email=email, error=str(e))
    if verify_voice(email, audio_bytes):
        login_time(email)
        user = User.query.filter_by(email=email).first()
        return render_template('home.html', account_number=user.account_number, user_type=user.user_type)
    return render_template('third.html', email=email, error="Voice verification failed.")

@app.route('/audio/<int:recording_id>')
def serve_audio(recording_id):
    """Serve a stored audio file by its recording ID."""
    recording = AudioRecording.query.get(recording_id)
    if recording and recording.raw_audio:
        return Response(recording.raw_audio, mimetype='audio/webm')
    return "Audio not found", 404

@app.route('/admin/users', methods=['GET'])
def list_users():
    users = User.query.all()
    return render_template('admin_users.html', users=users)

@app.route('/delete_user', methods=['POST'])
def delete_user():
    user_id = request.form.get('user_id')
    if not user_id:
        return "No user specified", 400
    user = User.query.filter_by(id=user_id).first()
    if user:
        db.session.delete(user)
        db.session.commit()
        return redirect(url_for('list_users'))
    return "User not found", 404

if __name__ == '__main__':
    app.run(debug=True)