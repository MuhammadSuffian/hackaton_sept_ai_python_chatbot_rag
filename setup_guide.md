# Supabase Data Fetcher Setup Guide

## Prerequisites
- Python 3.7 or higher
- A Supabase project

## Installation

1. **Install required packages:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Set up your Supabase credentials:**
   
   **Option A: Environment Variables (Recommended)**
   ```bash
   # Windows (PowerShell)
   $env:SUPABASE_URL="your_supabase_project_url"
   $env:SUPABASE_ANON_KEY="your_supabase_anon_key"
   
   # Windows (Command Prompt)
   set SUPABASE_URL=your_supabase_project_url
   set SUPABASE_ANON_KEY=your_supabase_anon_key
   
   # Linux/Mac
   export SUPABASE_URL="your_supabase_project_url"
   export SUPABASE_ANON_KEY="your_supabase_anon_key"
   ```
   
   **Option B: Direct Code Modification**
   Edit the `bd_fetch_test.py` file and replace:
   - `'your-supabase-url'` with your actual Supabase URL
   - `'your-supabase-anon-key'` with your actual Supabase anon key

## Getting Your Supabase Credentials

1. Go to your Supabase project dashboard
2. Navigate to Settings → API
3. Copy:
   - **Project URL** (for SUPABASE_URL)
   - **anon public** key (for SUPABASE_ANON_KEY)

## Usage

1. **Update the table name:**
   In `bd_fetch_test.py`, change `table_name = "your_table_name"` to your actual table name.

2. **Run the script:**
   ```bash
   python bd_fetch_test.py
   ```

## Features

- ✅ Fetch all data from a table
- ✅ Fetch limited number of records
- ✅ Fetch filtered data
- ✅ Beautiful formatted display
- ✅ Error handling
- ✅ Environment variable support

## Example Output

```
🚀 Supabase Data Fetcher
==================================================

📋 Fetching data from table: users

1️⃣ Fetching all data...

============================================================
📊 All data from users
============================================================
📈 Total records: 3
============================================================

🔹 Record 1:
----------------------------------------
  id: 1
  name: John Doe
  email: john@example.com
  created_at: 2024-01-01T00:00:00Z
----------------------------------------
```
