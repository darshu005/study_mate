import google.generativeai as genai

# ---------------------------------------------
# 🔹 STEP 1: CONFIGURE YOUR GEMINI API KEY
# ---------------------------------------------
# Replace YOUR_GEMINI_API_KEY_HERE with your real Gemini API key
# (You can find it at https://aistudio.google.com/app/apikey)
API_KEY = "AIzaSyCgYl7055D6Lyj-GFPsjRRwOlnVo7HfFoY"

if not API_KEY or API_KEY.strip() == "" or "YOUR_GEMINI_API_KEY_HERE" in API_KEY:
    print("❌ Please paste your actual Gemini API key inside the code first!")
    exit()

# Configure the client
genai.configure(api_key=API_KEY)

# ---------------------------------------------
# 🔹 STEP 2: LIST AVAILABLE MODELS
# ---------------------------------------------
print("\n🔍 Checking which Gemini models are available for your API key...\n")

try:
    models = genai.list_models()
    found_any = False

    for m in models:
        # Only show models that support text generation
        if "generateContent" in m.supported_generation_methods:
            print(f"✅ {m.name}  →  {m.supported_generation_methods}")
            found_any = True

    if not found_any:
        print("⚠ No models found that support text generation (generateContent).")
        print("Please verify your API key permissions in Google AI Studio.")

except Exception as e:
    print("\n❌ Error while listing models:")
    print(e)

# ---------------------------------------------
# 🔹 STEP 3: NEXT ACTION
# ---------------------------------------------
print("\n👉 Once you see a model name above like 'models/gemini-2.0-flash',")
print("   copy that name and update your StudyMate code like this:\n")
print("   model = genai.GenerativeModel('gemini-2.0-flash')\n")
