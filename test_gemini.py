import google.generativeai as genai

# Configure Gemini API
API_KEY = "AIzaSyBeg9Uh5g7rEJa42YjtU5uAQLwpgOMiXiA"
genai.configure(api_key=API_KEY)

try:
    # Create the model
    model = genai.GenerativeModel('gemini-1.5-flash')
    
    print("Testing Gemini API...")
    
    # Simple test prompt
    response = model.generate_content("Write a short description of Aloe-emodin, a compound found in Aloe vera.")
    
    print("✅ Success!")
    print("Response:", response.text)
    
except Exception as e:
    print("❌ Error:", e)
    print("Error type:", type(e).__name__)
