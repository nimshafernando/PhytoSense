@echo off
echo 🚀 Starting Medicinal Leaf Classifier Flask App with 3D Molecular Visualization...
echo.

echo 📋 Checking if required packages are installed...
python -c "import flask, torch, rdkit" 2>nul
if errorlevel 1 (
    echo ❌ Some required packages are missing. Installing...
    pip install -r flask_requirements.txt
    echo ✅ Packages installed!
) else (
    echo ✅ All packages are available!
)

echo.
echo 🌐 Starting Flask server...
echo 📝 The app will be available at: http://localhost:5000
echo 🛑 Press Ctrl+C to stop the server
echo.

python flask_app.py

echo.
echo 👋 Flask server stopped.
pause
