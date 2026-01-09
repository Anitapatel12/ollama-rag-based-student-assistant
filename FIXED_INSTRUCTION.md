# Fix Instructions for GenAI Document Assistant

## Issues Fixed:

### 1. ✅ TensorFlow Warnings
- Added environment variable `TF_ENABLE_ONEDNN_OPTS=0` to suppress oneDNN messages
- This reduces the TensorFlow logging noise

### 2. ✅ GPT4All Model Format Issue
- Updated code to expect `.gguf` format instead of old `.bin` format
- Created `requirements.txt` with proper dependencies

### 3. 🔄 GPT4All Model Download (Manual Step Required)

The current model `models/ggml-gpt4all-j-v1.3-groovy.bin` is in the old format. You need to download a compatible GGUF model:

#### Option A: Use Ollama (Recommended)
1. Install Ollama from https://ollama.ai/
2. Run: `ollama pull llama3`
3. Select "ollama" in the Streamlit app sidebar

#### Option B: Download GGUF Model Manually
1. Visit: https://gpt4all.io/
2. Download a GGUF model (e.g., "Mistral Instruct v0.2 GGUF")
3. Save it to `models/` directory
4. Update the model path in the app to match your downloaded file

#### Option C: Use Existing .bin Model (Temporary Fix)
If you want to use the existing .bin model temporarily, you can:
1. Rename the model file to `.gguf` (may not work with newer GPT4All versions)
2. Or downgrade GPT4All to an older version that supports .bin files

## Quick Start:

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Start the app:
   ```bash
   streamlit run app.py
   ```

3. Choose your LLM backend in the sidebar:
   - **Ollama**: Requires `ollama serve` running
   - **GPT4All**: Requires a GGUF model file

4. Upload a document and start using the features!

## Notes:
- TensorFlow warnings are now suppressed
- The app should run without the previous DLL loading errors
- GPT4All requires GGUF format models (not the old .bin format)
