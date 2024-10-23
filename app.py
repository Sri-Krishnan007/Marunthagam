from flask import Flask, render_template, request, redirect, url_for, flash,session, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_mail import Mail, Message
from werkzeug.security import generate_password_hash, check_password_hash
import re
import pandas as pd
import numpy as np
import pickle
import random
import string
app = Flask(__name__)
import streamlit as st
import subprocess
import os
import subprocess
from threading import Thread

# Configurations
app.config.from_object('config.Config')
db = SQLAlchemy(app)
mail = Mail(app)



# User model
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(150), nullable=False)
    email = db.Column(db.String(150), unique=True, nullable=False)
    password = db.Column(db.String(150), nullable=False)
    is_verified = db.Column(db.Boolean, default=False)
    verification_code = db.Column(db.String(6), nullable=True)

def run_streamlit():
    # Run Streamlit app as a separate process
    subprocess.run(["streamlit", "run", "chatbot.py"])

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        try:
            name = request.form['name']
            email = request.form['email']
            password = request.form['password']
            confirm_password = request.form['confirm_password']

            # Validate passwords match
            if password != confirm_password:
                flash("Passwords do not match", "danger")
                return redirect(url_for('signup'))

            # Validate password strength
            if len(password) < 8 or not re.search(r'[@$!%*?&#]', password):
                flash("Password must be at least 8 characters long and contain at least one special character", "danger")
                return redirect(url_for('signup'))

            # Check if email is already registered
            existing_user = User.query.filter_by(email=email).first()
            if existing_user:
                flash("Email is already registered", "danger")
                return redirect(url_for('signup'))

            # Hash the password and create user
            hashed_password = generate_password_hash(password, method='pbkdf2:sha256')
            verification_code = ''.join(random.choices(string.ascii_uppercase + string.digits, k=6))
            new_user = User(name=name, email=email, password=hashed_password, verification_code=verification_code)
            db.session.add(new_user)
            db.session.commit()

            # Send verification email
            msg = Message('Email Verification - Marunthagam', sender='srikrish2705guru@gmail.com', recipients=[email])
            msg.body = f"Thank you for registering! Use the following code to verify your email: {verification_code}"
            mail.send(msg)

            flash(f"Account created successfully. A verification email has been sent to {email}. Please check your inbox.", "success")

            # Redirect to the verification page
            return redirect(url_for('verify_email', email=email))  # Redirect to verification route
        
        except Exception as e:
            flash(f"An error occurred: {str(e)}", "danger")
            return redirect(url_for('signup'))

    return render_template('signup.html')

@app.route('/verify/<email>', methods=['GET', 'POST'])
def verify_email(email):
    user = User.query.filter_by(email=email).first()

    if request.method == 'POST':
        verification_code = request.form['verification_code']

        if user and user.verification_code == verification_code:
            user.is_verified = True
            user.verification_code = None  # Clear verification code after successful verification
            db.session.commit()
            flash("Email verified successfully! You can now log in.", "success")
            return redirect(url_for('index'))
        else:
            flash("Invalid verification code. Please try again.", "danger")

    return render_template('verify_email.html', email=email)


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        enable_geolocation = request.form.get('enable_geolocation')  # Checkbox value
        
        # Get user from DB
        user = User.query.filter_by(email=email).first()

        if user:
            # Verify if user is verified
            if not user.is_verified:
                flash("Your email is not verified. Please check your email for verification instructions.", "danger")
                return redirect(url_for('login'))
            
            # Check password
            if check_password_hash(user.password, password):
                # Generate OTP
                otp = ''.join(random.choices(string.digits, k=6))

                # Send OTP to user's email
                msg = Message('Your OTP Code', sender='srikrish2705guru@gmail.com', recipients=[email])
                msg.body = f"Your OTP code is: {otp}"
                mail.send(msg)

                # Store OTP in session (temporarily) without persisting to database
                session['otp'] = otp
                session['user_email'] = email

                # Redirect to OTP verification page
                flash("OTP has been sent to your email. Please verify.", "info")
                return redirect(url_for('verify_otp'))
            else:
                flash("Incorrect password. Please try again.", "danger")
        else:
            flash("No account found with that email.", "danger")

    return render_template('login.html')


@app.route('/verify-otp', methods=['GET', 'POST'])


def verify_otp():
    if request.method == 'POST':
        entered_otp = request.form['otp']

        # Check OTP
        if 'otp' in session and entered_otp == session['otp']:
            flash("Login successful!", "success")
            session.pop('otp', None)  # Remove OTP after successful login
            
            # Store user details in session
            user = User.query.filter_by(email=session['user_email']).first()
            session['user_name'] = user.name
            session['user_email'] = user.email
            
            # Run Streamlit app in a separate thread so it doesn't block Flask
            Thread(target=run_streamlit).start()
            
            # Redirect to Streamlit app with user details
            return redirect(f"http://localhost:8501/?name={user.name}&email={user.email}")
        else:
            flash("Invalid OTP. Please try again.", "danger")

    return render_template('verify_otp.html')





# Route for logout (already included earlier)
@app.route('/logout')
def logout():
    session.clear()  # Clear the session data
    return redirect(url_for('login'))

if __name__ == '__main__':
    app.run(debug=True)
    
    
    
    



    
    
