from datetime import datetime
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.dialects.mysql import LONGTEXT
db = SQLAlchemy()

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(120), nullable=False)
    # NEW: Add a status column for admin approval
    status = db.Column(db.String(20), default='pending', nullable=False)
    # Optional: Add a role column to differentiate admins from regular users
    role = db.Column(db.String(20), default='user', nullable=False)

class Log(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    
    # Use ForeignKey to link to the User who performed the action
    user_name = db.Column(db.String(80), nullable=False)
    
    # Store the initial user query (can be long, so use Text)
    user_query = db.Column(db.Text, nullable=False)
    
    # Store the JSON system response as a string
    system_response = db.Column(LONGTEXT, nullable=False)
    
    # Store the feedback text
    feedback = db.Column(db.String(255), nullable=True)
    
    # Automatically set the timestamp when a log is created
    timestamp = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
