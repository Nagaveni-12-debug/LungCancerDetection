from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify
import os
from werkzeug.utils import secure_filename
import sqlite3
from datetime import datetime
import predict_model

app = Flask(__name__)
app.secret_key = 'cancer_detection_secret_key_2024'
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'dcm'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def init_db():
    """Initialize the database with required tables"""
    conn = sqlite3.connect('database.db')
    c = conn.cursor()
    
    # Users table
    c.execute('''CREATE TABLE IF NOT EXISTS users
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  username TEXT UNIQUE NOT NULL,
                  email TEXT UNIQUE NOT NULL,
                  password TEXT NOT NULL,
                  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')
    
    # Scans table
    c.execute('''CREATE TABLE IF NOT EXISTS scans
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  user_id INTEGER NOT NULL,
                  filename TEXT NOT NULL,
                  result TEXT NOT NULL,
                  confidence REAL NOT NULL,
                  cancer_prob REAL NOT NULL,
                  healthy_prob REAL NOT NULL,
                  tumor_location TEXT,
                  tumor_size REAL,
                  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                  FOREIGN KEY (user_id) REFERENCES users (id))''')
    
    conn.commit()
    conn.close()

def get_db_connection():
    conn = sqlite3.connect('database.db')
    conn.row_factory = sqlite3.Row
    return conn

# Initialize database
init_db()

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']
        confirm_password = request.form['confirm_password']
        
        if password != confirm_password:
            flash('Passwords do not match!', 'error')
            return render_template('register.html')
        
        conn = get_db_connection()
        try:
            conn.execute('INSERT INTO users (username, email, password) VALUES (?, ?, ?)',
                        (username, email, password))
            conn.commit()
            flash('Registration successful! Please login.', 'success')
            return redirect(url_for('login'))
        except sqlite3.IntegrityError:
            flash('Username or email already exists!', 'error')
        finally:
            conn.close()
    
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        conn = get_db_connection()
        user = conn.execute('SELECT * FROM users WHERE username = ? AND password = ?',
                           (username, password)).fetchone()
        conn.close()
        
        if user:
            session['user_id'] = user['id']
            session['username'] = user['username']
            flash('Login successful!', 'success')
            return redirect(url_for('dashboard'))
        else:
            flash('Invalid username or password!', 'error')
    
    return render_template('login.html')

@app.route('/dashboard')
def dashboard():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    conn = get_db_connection()
    
    # Get user statistics
    total_scans = conn.execute('SELECT COUNT(*) FROM scans WHERE user_id = ?', 
                              (session['user_id'],)).fetchone()[0]
    
    cancer_cases = conn.execute('SELECT COUNT(*) FROM scans WHERE user_id = ? AND result = "Cancer"', 
                               (session['user_id'],)).fetchone()[0]
    
    healthy_cases = conn.execute('SELECT COUNT(*) FROM scans WHERE user_id = ? AND result = "Healthy"', 
                                (session['user_id'],)).fetchone()[0]
    
    # Get recent scans
    recent_scans = conn.execute('''
        SELECT * FROM scans 
        WHERE user_id = ? 
        ORDER BY created_at DESC 
        LIMIT 5
    ''', (session['user_id'],)).fetchall()
    
    conn.close()
    
    return render_template('dashboard.html',
                         username=session['username'],
                         total_scans=total_scans,
                         cancer_cases=cancer_cases,
                         healthy_cases=healthy_cases,
                         recent_scans=recent_scans)

@app.route('/predict')
def predict():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    return render_template('predict.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    if 'user_id' not in session:
        return jsonify({'success': False, 'error': 'Not logged in'})
    
    if 'file' not in request.files:
        return jsonify({'success': False, 'error': 'No file uploaded'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'success': False, 'error': 'No file selected'})
    
    if file and allowed_file(file.filename):
        try:
            # Create uploads directory if it doesn't exist
            os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
            
            # Save the file
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Analyze the image using your model
            result = predict_model.predict_image(filepath)
            
            if result and isinstance(result, dict):
                # Store results in database
                conn = get_db_connection()
                conn.execute('''
                    INSERT INTO scans (user_id, filename, result, confidence, cancer_prob, healthy_prob, tumor_location, tumor_size)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (session['user_id'], filename, result['result'], result['confidence'], 
                      result['cancer_prob'], result['healthy_prob'], 
                      result['tumor_location'], result['tumor_size']))
                conn.commit()
                conn.close()
                
                # Store results in session for results page
                session['analysis_result'] = result
                return jsonify({'success': True})
            else:
                return jsonify({'success': False, 'error': 'Analysis failed'})
                
        except Exception as e:
            return jsonify({'success': False, 'error': str(e)})
    
    return jsonify({'success': False, 'error': 'Invalid file type'})

@app.route('/results')
def results():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    result = session.get('analysis_result')
    if not result:
        return redirect(url_for('predict'))
    
    return render_template('results.html', 
                         confidence_val=result['confidence'],
                         cancer_prob=result['cancer_prob'],
                         healthy_prob=result['healthy_prob'],
                         tuner_location=result['tumor_location'],
                         tuner_size=result['tumor_size'])

@app.route('/history')
def history():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    conn = get_db_connection()
    scans = conn.execute('''
        SELECT * FROM scans 
        WHERE user_id = ? 
        ORDER BY created_at DESC
    ''', (session['user_id'],)).fetchall()
    conn.close()
    
    return render_template('history.html', scans=scans)

@app.route('/admin')
def admin():
    # Simple admin check - in production, use proper authentication
    if 'user_id' not in session or session.get('username') != 'admin':
        return redirect(url_for('dashboard'))
    
    conn = get_db_connection()
    
    # Get admin statistics
    total_users = conn.execute('SELECT COUNT(*) FROM users').fetchone()[0]
    total_scans = conn.execute('SELECT COUNT(*) FROM scans').fetchone()[0]
    cancer_cases = conn.execute('SELECT COUNT(*) FROM scans WHERE result = "Cancer"').fetchone()[0]
    
    # Get all users
    users = conn.execute('SELECT * FROM users ORDER BY created_at DESC').fetchall()
    
    conn.close()
    
    return render_template('admin.html',
                         total_users=total_users,
                         total_scans=total_scans,
                         cancer_cases=cancer_cases,
                         users=users)

@app.route('/logout')
def logout():
    session.clear()
    flash('You have been logged out successfully!', 'success')
    return redirect(url_for('index'))

if __name__ == '__main__':
    # Create uploads directory if it doesn't exist
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True, host='0.0.0.0', port=5000)