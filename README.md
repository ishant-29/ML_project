# Smart Clothing Size Predictor

An AI-powered web application that predicts clothing sizes based on user measurements and preferences.

## Features

- AI-powered size prediction
- Support for multiple brands (Adidas, H&M, Nike, Uniqlo, Zara)
- Support for multiple item types (Dress, Jacket, Jeans, Shirt, T-Shirt)
- Modern, responsive UI with animations
- Real-time form validation and progress tracking

## Deployment on PythonAnywhere

### Step 1: Create PythonAnywhere Account
1. Go to [www.pythonanywhere.com](https://www.pythonanywhere.com)
2. Sign up for a free account

### Step 2: Upload Your Files

#### Option A: Using Git (Recommended)
1. Push your code to GitHub/GitLab
2. In PythonAnywhere, go to "Consoles" → "Bash"
3. Clone your repository:
```bash
git clone https://github.com/yourusername/your-repo-name.git
cd your-repo-name/Size_predictor
```

#### Option B: Manual Upload
1. Go to "Files" in PythonAnywhere
2. Create a new directory for your project
3. Upload all files from the `Size_predictor` folder

### Step 3: Set Up Virtual Environment
1. Go to "Consoles" → "Bash"
2. Navigate to your project directory
3. Create and activate virtual environment:
```bash
cd your-project-directory
python3 -m venv venv
source venv/bin/activate
```

### Step 4: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 5: Configure Web App
1. Go to "Web" tab in PythonAnywhere
2. Click "Add a new web app"
3. Choose "Flask" and Python 3.9 (or latest available)
4. Set the path to your project directory

### Step 6: Configure WSGI File
1. Click on the WSGI configuration file link
2. Replace the content with:
```python
import sys
import os

# Add your project directory to Python path
sys.path.insert(0, '/home/yourusername/your-project-directory')

from main import app as application
```

### Step 7: Set Up Static Files
1. In the "Web" tab, scroll down to "Static files"
2. Add these configurations:
   - URL: `/static/`
   - Directory: `/home/yourusername/your-project-directory/static`

### Step 8: Reload Web App
1. Click the "Reload" button in the "Web" tab
2. Your app should now be accessible at `yourusername.pythonanywhere.com`

## File Structure
```
Size_predictor/
├── main.py              # Flask application
├── wsgi.py              # WSGI configuration for PythonAnywhere
├── requirements.txt     # Python dependencies
├── model.pkl           # Trained ML model
├── brand_encoder.pkl   # Brand label encoder
├── item_encoder.pkl    # Item type encoder
├── scaler.pkl          # Feature scaler
├── selector.pkl        # Feature selector
├── Cloth_dataset.csv   # Training dataset
├── static/             # Static files (CSS, JS)
│   ├── css/
│   │   └── style.css
│   └── js/
│       └── script.js
└── templates/          # HTML templates
    ├── index.html
    └── prediction_result.html
```

## Troubleshooting

### Common Issues:

1. **Import Errors**: Make sure all `.pkl` files are in the same directory as `main.py`

2. **Static Files Not Loading**: Check that static file paths are correctly configured in PythonAnywhere

3. **Model Loading Errors**: Ensure all model files are uploaded and have correct permissions

4. **500 Internal Server Error**: Check the error logs in PythonAnywhere's "Web" tab

### Error Logs:
- Go to "Web" tab → "Error log" to see detailed error messages
- Check "Server log" for additional debugging information

## Local Development

To run locally:
```bash
pip install -r requirements.txt
python main.py
```

The app will be available at `http://localhost:5000`

## Technologies Used

- **Backend**: Flask, Python
- **Machine Learning**: scikit-learn, joblib
- **Frontend**: HTML5, CSS3, JavaScript
- **UI/UX**: Custom CSS animations, responsive design
- **Deployment**: PythonAnywhere 