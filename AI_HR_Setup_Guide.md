# 🤖 AI HR Assistant Setup Guide

## Overview
This AI HR Assistant uses RAG (Retrieval-Augmented Generation) to answer questions about your HR data stored in Supabase. It can fetch and analyze data from `organizations` and `employees` tables.

## Features

### 🧠 AI-Powered RAG System
- **Semantic Search**: Uses sentence transformers for intelligent content retrieval
- **Context-Aware Responses**: Answers questions based on your actual HR data
- **Similarity Matching**: Finds relevant information using cosine similarity

### 📊 Data Management
- **Organizations Table**: Fetches company data (name, domain)
- **Employees Table**: Fetches employee data (name, email, role, phone, organization_id)
- **Real-time Updates**: Always works with latest data from Supabase

### 💬 Interactive Chat Interface
- **Natural Language Queries**: Ask questions in plain English
- **Chat History**: Maintains conversation context
- **Sample Questions**: Pre-built queries to get started

### 📈 HR Insights Dashboard
- **Employee Metrics**: Total count, roles, organizations
- **Data Visualization**: Interactive tables and metrics
- **Real-time Statistics**: Live data analysis

## Installation

1. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the AI HR Assistant:**
   ```bash
   streamlit run ai_hr_assistant.py
   ```

3. **Open Browser:**
   Navigate to `http://localhost:8501`

## Usage

### 1. Load Data
- Click "🔄 Load All Data" in the sidebar
- Wait for data to be fetched from Supabase
- AI model will generate embeddings for RAG

### 2. Ask Questions
Try these sample questions:
- "How many employees do we have?"
- "List all employees"
- "What roles are available?"
- "Show me all organizations"
- "Who works in organization 1?"
- "What are the contact details?"

### 3. View Insights
- Check the HR Insights panel for metrics
- View data tables in the sidebar
- Analyze employee and organization statistics

## Database Schema

### Organizations Table
```sql
- id (Primary Key)
- name (Organization Name)
- domain (Organization Domain)
```

### Employees Table
```sql
- id (Primary Key)
- organization_id (Foreign Key to Organizations)
- name (Employee Name)
- email (Employee Email)
- role (Employee Role)
- phone (Employee Phone)
```

## AI Capabilities

### 🤖 Natural Language Processing
- Understands HR-related queries
- Extracts relevant information from your data
- Provides contextual responses

### 🔍 Intelligent Search
- Semantic similarity matching
- Context-aware retrieval
- Multi-table data correlation

### 📊 Data Analysis
- Employee statistics
- Role distribution
- Organization insights
- Contact information management

## Sample Queries

### Employee Queries
- "How many employees are there?"
- "Show me all employees"
- "Who has the role of manager?"
- "What are all the available roles?"
- "Show me employees from organization 1"

### Organization Queries
- "How many organizations do we have?"
- "List all organizations"
- "What domains are we working with?"
- "Show me organization details"

### Contact Queries
- "What are the email addresses?"
- "Show me phone numbers"
- "List all contact information"

## Technical Details

### RAG Implementation
- **Embedding Model**: `all-MiniLM-L6-v2` (sentence-transformers)
- **Similarity**: Cosine similarity for relevance scoring
- **Chunking**: Automatic text chunking from database records
- **Retrieval**: Top-k similarity search

### Performance
- **Fast Embeddings**: Optimized sentence transformer model
- **Efficient Search**: Vector similarity with scikit-learn
- **Real-time**: Live data fetching from Supabase

## Troubleshooting

### Common Issues
1. **Model Loading**: First run may take time to download the AI model
2. **Data Loading**: Ensure Supabase credentials are correct
3. **Memory**: Large datasets may require more RAM

### Solutions
- Check Supabase connection in sidebar
- Verify table names match your database
- Ensure all dependencies are installed

## Advanced Features

### Custom Queries
You can ask complex questions like:
- "Show me all managers and their contact details"
- "Which organizations have the most employees?"
- "List employees by role and organization"

### Data Export
- View raw data in JSON format
- Export to CSV through Streamlit interface
- Copy data for external analysis

## Support

For issues or questions:
1. Check the console for error messages
2. Verify Supabase connection
3. Ensure all dependencies are installed
4. Check that your database has the required tables

---

**Ready to revolutionize your HR data management with AI! 🚀**
