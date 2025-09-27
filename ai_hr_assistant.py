import os
import streamlit as st
import pandas as pd
from supabase import create_client, Client
from typing import List, Dict, Any
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import re
from datetime import datetime

# Configure Streamlit page
st.set_page_config(
    page_title="AI HR Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

class AIHRAssistant:
    def __init__(self):
        """Initialize AI HR Assistant with Supabase connection and RAG capabilities"""
        # Supabase configuration
        self.url = os.getenv('SUPABASE_URL', 'https://ptkqgiqqefoceswfwent.supabase.co')
        self.key = os.getenv('SUPABASE_ANON_KEY', 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InB0a3FnaXFxZWZvY2Vzd2Z3ZW50Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTg5Njk1NDcsImV4cCI6MjA3NDU0NTU0N30.6DU3Ahplr8tybCxBcJScd8PaPqYWXNb_Y7zSMvF3wRg')
        
        self.supabase: Client = create_client(self.url, self.key)
        
        # Initialize RAG components
        self.embedding_model = None
        self.organizations_data = []
        self.employees_data = []
        self.embeddings = None
        self.text_chunks = []
        
        # Load embedding model
        self._load_embedding_model()
    
    def _load_embedding_model(self):
        """Load the sentence transformer model for embeddings"""
        try:
            with st.spinner("Loading AI model..."):
                self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            st.success("✅ AI model loaded successfully!")
        except Exception as e:
            st.error(f"❌ Error loading AI model: {str(e)}")
    
    def fetch_organizations(self) -> List[Dict[Any, Any]]:
        """Fetch all organizations data"""
        try:
            response = self.supabase.table('organizations').select("*").execute()
            self.organizations_data = response.data
            return response.data
        except Exception as e:
            st.error(f"❌ Error fetching organizations: {str(e)}")
            return []
    
    def fetch_employees(self) -> List[Dict[Any, Any]]:
        """Fetch all employees data"""
        try:
            response = self.supabase.table('employees').select("*").execute()
            self.employees_data = response.data
            return response.data
        except Exception as e:
            st.error(f"❌ Error fetching employees: {str(e)}")
            return []
    
    def fetch_employee_by_organization(self, org_id: int) -> List[Dict[Any, Any]]:
        """Fetch employees by organization ID"""
        try:
            response = self.supabase.table('employees').select("*").eq('organization_id', org_id).execute()
            return response.data
        except Exception as e:
            st.error(f"❌ Error fetching employees by organization: {str(e)}")
            return []
    
    def prepare_rag_data(self):
        """Prepare data for RAG system by creating text chunks and embeddings"""
        if not self.embedding_model:
            st.error("❌ AI model not loaded!")
            return
        
        self.text_chunks = []
        self.embeddings = []
        
        # Create text chunks from organizations
        for org in self.organizations_data:
            chunk = f"Organization: {org.get('name', 'Unknown')} - Domain: {org.get('domain', 'Unknown')}"
            self.text_chunks.append(chunk)
        
        # Create text chunks from employees
        for emp in self.employees_data:
            # Find organization name
            org_name = "Unknown"
            for org in self.organizations_data:
                if org.get('id') == emp.get('organization_id'):
                    org_name = org.get('name', 'Unknown')
                    break
            
            chunk = f"Employee: {emp.get('name', 'Unknown')} - Email: {emp.get('email', 'Unknown')} - Role: {emp.get('role', 'Unknown')} - Phone: {emp.get('phone', 'Unknown')} - Organization: {org_name}"
            self.text_chunks.append(chunk)
        
        # Generate embeddings
        if self.text_chunks:
            with st.spinner("Generating embeddings..."):
                self.embeddings = self.embedding_model.encode(self.text_chunks)
            st.success(f"✅ Generated {len(self.embeddings)} embeddings!")
    
    def search_similar_content(self, query: str, top_k: int = 5) -> List[tuple]:
        """Search for similar content using RAG"""
        if not self.embedding_model or self.embeddings is None or len(self.embeddings) == 0:
            return []
        
        # Generate query embedding
        query_embedding = self.embedding_model.encode([query])
        
        # Calculate similarities
        similarities = cosine_similarity(query_embedding, self.embeddings)[0]
        
        # Get top-k most similar chunks
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            if similarities[idx] > 0.1:  # Threshold for relevance
                results.append((self.text_chunks[idx], similarities[idx]))
        
        return results
    
    def answer_hr_question(self, question: str) -> str:
        """Answer HR questions using RAG"""
        # Check if data is loaded
        if not self.organizations_data and not self.employees_data:
            return "❌ No data loaded. Please click 'Load All Data' in the sidebar first."
        
        # Search for relevant information
        relevant_chunks = self.search_similar_content(question, top_k=3)
        
        if not relevant_chunks:
            return "I couldn't find relevant information to answer your question. Please make sure the data is loaded and try rephrasing your question."
        
        # Generate answer based on relevant chunks
        context = "\n".join([chunk[0] for chunk in relevant_chunks])
        
        # Simple rule-based responses for common HR questions
        question_lower = question.lower()
        
        if any(word in question_lower for word in ['employee', 'employees', 'staff', 'team']):
            if 'count' in question_lower or 'how many' in question_lower:
                return f"📊 **Employee Count**: {len(self.employees_data)} employees found in the database."
            elif 'list' in question_lower or 'show' in question_lower:
                return f"📋 **Employee List**: Found {len(self.employees_data)} employees. Here are the details:\n\n{context}"
        
        elif any(word in question_lower for word in ['organization', 'company', 'companies']):
            if 'count' in question_lower or 'how many' in question_lower:
                return f"🏢 **Organization Count**: {len(self.organizations_data)} organizations found."
            elif 'list' in question_lower or 'show' in question_lower:
                return f"🏢 **Organization List**: Found {len(self.organizations_data)} organizations. Here are the details:\n\n{context}"
        
        elif any(word in question_lower for word in ['role', 'roles', 'position']):
            roles = set(emp.get('role', 'Unknown') for emp in self.employees_data)
            return f"👔 **Available Roles**: {', '.join(roles)}"
        
        elif any(word in question_lower for word in ['email', 'contact']):
            return f"📧 **Contact Information**: Here are the contact details:\n\n{context}"
        
        else:
            return f"🤖 **AI HR Assistant Response**:\n\nBased on the available data, here's what I found:\n\n{context}\n\n*Relevance scores: {[f'{chunk[1]:.2f}' for chunk in relevant_chunks]}*"
    
    def get_hr_insights(self) -> Dict[str, Any]:
        """Generate HR insights from the data"""
        insights = {}
        
        if self.employees_data:
            # Employee statistics
            insights['total_employees'] = len(self.employees_data)
            insights['roles'] = list(set(emp.get('role', 'Unknown') for emp in self.employees_data))
            insights['organizations_with_employees'] = len(set(emp.get('organization_id') for emp in self.employees_data))
        
        if self.organizations_data:
            insights['total_organizations'] = len(self.organizations_data)
            insights['domains'] = list(set(org.get('domain', 'Unknown') for org in self.organizations_data))
        
        return insights

def main():
    """Main Streamlit app"""
    st.title("🤖 AI HR Assistant")
    st.markdown("---")
    
    # Initialize the AI HR Assistant
    if 'hr_assistant' not in st.session_state:
        st.session_state.hr_assistant = AIHRAssistant()
    
    hr_assistant = st.session_state.hr_assistant
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Data Management")
        
        if st.button("🔄 Load All Data", type="primary"):
            with st.spinner("Loading data..."):
                organizations = hr_assistant.fetch_organizations()
                employees = hr_assistant.fetch_employees()
                
                if organizations and employees:
                    hr_assistant.prepare_rag_data()
                    st.success("✅ Data loaded successfully!")
                else:
                    st.error("❌ Failed to load data!")
        
        st.subheader("📊 Data Status")
        st.info(f"Organizations: {len(hr_assistant.organizations_data)}")
        st.info(f"Employees: {len(hr_assistant.employees_data)}")
        
        # Data display options
        st.subheader("📋 View Data")
        if st.button("View Organizations"):
            if hr_assistant.organizations_data:
                df_org = pd.DataFrame(hr_assistant.organizations_data)
                st.dataframe(df_org)
            else:
                st.warning("No organizations data loaded")
        
        if st.button("View Employees"):
            if hr_assistant.employees_data:
                df_emp = pd.DataFrame(hr_assistant.employees_data)
                st.dataframe(df_emp)
            else:
                st.warning("No employees data loaded")
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("💬 Chat with AI HR Assistant")
        
        # Chat interface
        if "messages" not in st.session_state:
            st.session_state.messages = []
        
        # Display chat history
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        # Chat input
        if prompt := st.chat_input("Ask me anything about your HR data..."):
            # Add user message
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Generate AI response
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    response = hr_assistant.answer_hr_question(prompt)
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
    
    with col2:
        st.header("📈 HR Insights")
        
        if hr_assistant.employees_data or hr_assistant.organizations_data:
            insights = hr_assistant.get_hr_insights()
            
            st.metric("Total Employees", insights.get('total_employees', 0))
            st.metric("Total Organizations", insights.get('total_organizations', 0))
            st.metric("Organizations with Employees", insights.get('organizations_with_employees', 0))
            
            if insights.get('roles'):
                st.subheader("👔 Roles")
                for role in insights['roles']:
                    st.write(f"• {role}")
            
            if insights.get('domains'):
                st.subheader("🌐 Domains")
                for domain in insights['domains']:
                    st.write(f"• {domain}")
        else:
            st.info("Load data to see insights")
        
        # Connection status
        st.subheader("🔗 Connection Status")
        try:
            # Test connection by trying to fetch from organizations table
            test_response = hr_assistant.supabase.table("organizations").select("*").limit(1).execute()
            st.success("🟢 Connected to Supabase")
        except Exception as e:
            st.error(f"🔴 Connection failed: {str(e)}")
        
        # Sample questions
        st.subheader("💡 Sample Questions")
        sample_questions = [
            "How many employees do we have?",
            "List all employees",
            "What roles are available?",
            "Show me all organizations",
            "Who works in organization 1?",
            "What are the contact details?"
        ]
        
        for question in sample_questions:
            if st.button(question, key=f"sample_{question}"):
                st.session_state.messages.append({"role": "user", "content": question})
                with st.spinner("Thinking..."):
                    response = hr_assistant.answer_hr_question(question)
                st.session_state.messages.append({"role": "assistant", "content": response})
                st.rerun()

if __name__ == "__main__":
    main()
