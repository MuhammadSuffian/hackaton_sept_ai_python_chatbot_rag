# AI HR Assistant Configuration Guide

## Environment Variables

Create a `.env` file in your project root with the following variables:

### Required Variables:
```bash
# Supabase Configuration
SUPABASE_URL=your_supabase_url_here
SUPABASE_ANON_KEY=your_supabase_anon_key_here

# Groq API Configuration
GROQ_API_KEY=your_groq_api_key_here
```

### Optional Variables (with defaults):
```bash
# Groq Model Configuration
GROQ_MODEL=openai/gpt-oss-120b
GROQ_TEMPERATURE=0.1
GROQ_MAX_TOKENS=1000
```

## Setup Instructions:

1. **Get Groq API Key:**
   - Visit https://console.groq.com/
   - Sign up/login to your account
   - Generate an API key
   - Add it to your `.env` file as `GROQ_API_KEY`

2. **Configure Supabase:**
   - Use your existing Supabase URL and anon key
   - Add them to your `.env` file

3. **Model Options:**
   - `openai/gpt-oss-120b` (default, fast)
   - `llama3-8b-8192` (alternative)
   - `llama3-70b-8192` (more powerful)

4. **Run the application:**
   ```bash
   streamlit run ai_hr_assistant.py
   ```

## Features:
- ✅ No hardcoded API keys
- ✅ Configurable model selection
- ✅ Environment-based configuration
- ✅ Graceful fallback if Groq is not configured
