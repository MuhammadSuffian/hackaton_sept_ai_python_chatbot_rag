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

# Groq API integration
from groq import Groq
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document

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
        
        # Initialize Groq client
        groq_api_key = "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"
        self.groq_client = Groq(api_key=groq_api_key)
        
        # Initialize RAG components
        self.embedding_model = None
        self.organizations_data = []
        self.employees_data = []
        self.embeddings = None
        self.text_chunks = []
        self.vectorstore = None
        
        # Load embedding model
        self._load_embedding_model()
    
    def _load_embedding_model(self):
        """Load the sentence transformer model for embeddings"""
        try:
            with st.spinner("Loading AI model..."):
                self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            st.success("✅ AI model loaded successfully!")
            
            # Automatically load data after model loads
            self._auto_load_data()
        except Exception as e:
            st.error(f"❌ Error loading AI model: {str(e)}")
    
    def _auto_load_data(self):
        """Automatically load all database data after AI model loads"""
        try:
            with st.spinner("Loading database data..."):
                # Fetch organizations and employees
                organizations = self.fetch_organizations()
                employees = self.fetch_employees()
                
                if organizations and employees:
                    # Prepare RAG data
                    self.prepare_rag_data()
                    st.success("✅ All data loaded successfully!")
                else:
                    st.warning("⚠️ No data found in database")
        except Exception as e:
            st.error(f"❌ Error loading data: {str(e)}")
    
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
        documents = []
        
        # Create text chunks from organizations
        for org in self.organizations_data:
            chunk = f"Organization: {org.get('name', 'Unknown')} - Domain: {org.get('domain', 'Unknown')}"
            self.text_chunks.append(chunk)
            documents.append(Document(page_content=chunk, metadata={"type": "organization", "id": org.get('id')}))
        
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
            documents.append(Document(page_content=chunk, metadata={"type": "employee", "id": emp.get('id')}))
        
        # Create FAISS vectorstore
        if documents:
            with st.spinner("Creating vectorstore..."):
                embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
                self.vectorstore = FAISS.from_documents(documents, embeddings)
            st.success(f"✅ Created vectorstore with {len(documents)} documents!")
    
    def search_similar_content(self, query: str, top_k: int = 5) -> List[tuple]:
        """Search for similar content using FAISS vectorstore"""
        if not self.vectorstore:
            return []
        
        try:
            # Search using FAISS vectorstore
            docs = self.vectorstore.similarity_search_with_score(query, k=top_k)
            results = []
            for doc, score in docs:
                # Convert FAISS distance to similarity (lower distance = higher similarity)
                similarity = 1.0 / (1.0 + score)  # Convert distance to similarity
                if similarity > 0.1:  # Threshold for relevance
                    results.append((doc.page_content, similarity))
            return results
        except Exception as e:
            st.error(f"❌ Error in similarity search: {str(e)}")
            return []
    
    def answer_hr_question(self, question: str) -> str:
        """Answer HR questions using RAG with enhanced analytical capabilities"""
        question_lower = question.lower().strip()
        
        
        # Check if data is loaded for HR-specific queries first
        if not self.organizations_data and not self.employees_data:
            return "❌ No HR data loaded. Please wait for automatic data loading to complete. I can still help with general questions though!"
        
        # Enhanced analytical queries (check these BEFORE greetings)
        if any(phrase in question_lower for phrase in ['organizations with most employees', 'companies with most staff', 'organizations by employee count', 'which organization has highest employee', 'which organization has most employees', 'highest employee organization', 'which organization has most employees', 'organization with most employees', 'most employees organization', 'organization most employees', 'most employees', 'highest employees', 'organization has most']):
            return self._get_organizations_by_employee_count()
        
        elif any(phrase in question_lower for phrase in ['employees by organization', 'staff by company', 'organization employee breakdown']):
            return self._get_employee_breakdown_by_organization()
        
        elif any(phrase in question_lower for phrase in ['roles by organization', 'positions by company', 'job roles breakdown']):
            return self._get_roles_by_organization()
        
        elif any(phrase in question_lower for phrase in ['largest organization', 'biggest company', 'organization with most people', 'which organization is largest', 'which company is biggest', 'which organization has most', 'which company has most', 'most employees', 'highest number of employees']):
            return self._get_largest_organization()
        
        elif any(phrase in question_lower for phrase in ['smallest organization', 'smallest company', 'organization with least people', 'which organization is smallest']):
            return self._get_smallest_organization()
        
        # Handle greetings and general conversation (after analytical queries)
        elif any(greeting in question_lower for greeting in ['hello', 'hi', 'hey', 'good morning', 'good afternoon', 'good evening']):
            return self._handle_greeting()
        
        elif any(phrase in question_lower for phrase in ['how are you', 'how do you do', 'what can you do', 'what are your capabilities']):
            return self._handle_capabilities_question()
        
        elif any(phrase in question_lower for phrase in ['help', 'what can you help with', 'assist']):
            return self._handle_help_question()
        
        elif any(phrase in question_lower for phrase in ['thank you', 'thanks', 'appreciate']):
            return self._handle_thanks()
        
        # Check if this is a general non-HR question first
        if self._is_general_question(question_lower):
            return self._handle_general_question(question)
        
        # Search for relevant information using RAG
        relevant_chunks = self.search_similar_content(question, top_k=3)
        
        if not relevant_chunks:
            return self._handle_general_question(question)
        
        # Generate answer based on relevant chunks
        context = "\n".join([chunk[0] for chunk in relevant_chunks])
        
        # Use Groq for enhanced responses (only for HR-related queries)
        try:
            return self._generate_groq_response(question, context)
        except Exception as e:
            st.error(f"❌ Error with Groq API: {str(e)}")
            # Fallback to rule-based responses
            return self._fallback_response(question_lower, context, relevant_chunks)
    
    def _fallback_response(self, question_lower: str, context: str, relevant_chunks: List[tuple]) -> str:
        """Fallback response when Groq API fails"""
        # Simple rule-based responses for common HR questions
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
    
    def _get_organizations_by_employee_count(self) -> str:
        """Get organizations ranked by employee count (descending order)"""
        if not self.employees_data or not self.organizations_data:
            return "❌ No data available for analysis."
        
        # Count employees per organization
        org_employee_count = {}
        for emp in self.employees_data:
            org_id = emp.get('organization_id')
            if org_id:
                org_employee_count[org_id] = org_employee_count.get(org_id, 0) + 1
        
        # Create list of (org_name, employee_count) tuples
        org_stats = []
        for org in self.organizations_data:
            org_id = org.get('id')
            org_name = org.get('name', 'Unknown')
            employee_count = org_employee_count.get(org_id, 0)
            org_stats.append((org_name, employee_count))
        
        # Sort by employee count (descending)
        org_stats.sort(key=lambda x: x[1], reverse=True)
        
        # Format response
        result = "🏢 **Organizations by Employee Count** (Most to Least):\n\n"
        for i, (org_name, count) in enumerate(org_stats, 1):
            result += f"{i}. **{org_name}**: {count} employees\n"
        
        return result
    
    def _get_employee_breakdown_by_organization(self) -> str:
        """Get detailed employee breakdown by organization"""
        if not self.employees_data or not self.organizations_data:
            return "❌ No data available for analysis."
        
        # Group employees by organization
        org_employees = {}
        for emp in self.employees_data:
            org_id = emp.get('organization_id')
            if org_id:
                if org_id not in org_employees:
                    org_employees[org_id] = []
                org_employees[org_id].append(emp)
        
        result = "📊 **Employee Breakdown by Organization**:\n\n"
        
        for org in self.organizations_data:
            org_id = org.get('id')
            org_name = org.get('name', 'Unknown')
            employees = org_employees.get(org_id, [])
            
            result += f"🏢 **{org_name}** ({len(employees)} employees):\n"
            for emp in employees:
                result += f"  • {emp.get('name', 'Unknown')} - {emp.get('role', 'Unknown')}\n"
            result += "\n"
        
        return result
    
    def _get_roles_by_organization(self) -> str:
        """Get roles breakdown by organization"""
        if not self.employees_data or not self.organizations_data:
            return "❌ No data available for analysis."
        
        # Group roles by organization
        org_roles = {}
        for emp in self.employees_data:
            org_id = emp.get('organization_id')
            role = emp.get('role', 'Unknown')
            if org_id:
                if org_id not in org_roles:
                    org_roles[org_id] = set()
                org_roles[org_id].add(role)
        
        result = "👔 **Roles by Organization**:\n\n"
        
        for org in self.organizations_data:
            org_id = org.get('id')
            org_name = org.get('name', 'Unknown')
            roles = org_roles.get(org_id, set())
            
            result += f"🏢 **{org_name}**:\n"
            for role in sorted(roles):
                result += f"  • {role}\n"
            result += "\n"
        
        return result
    
    def _get_largest_organization(self) -> str:
        """Get the organization with the most employees"""
        if not self.employees_data or not self.organizations_data:
            return "❌ No data available for analysis."
        
        # Count employees per organization
        org_employee_count = {}
        for emp in self.employees_data:
            org_id = emp.get('organization_id')
            if org_id:
                org_employee_count[org_id] = org_employee_count.get(org_id, 0) + 1
        
        if not org_employee_count:
            return "❌ No employees found in any organization."
        
        # Find organization with most employees
        max_org_id = max(org_employee_count, key=org_employee_count.get)
        max_count = org_employee_count[max_org_id]
        
        # Get organization name
        org_name = "Unknown"
        for org in self.organizations_data:
            if org.get('id') == max_org_id:
                org_name = org.get('name', 'Unknown')
                break
        
        return f"🏆 **Largest Organization**: **{org_name}** with {max_count} employees"
    
    def _get_smallest_organization(self) -> str:
        """Get the organization with the least employees"""
        if not self.employees_data or not self.organizations_data:
            return "❌ No data available for analysis."
        
        # Count employees per organization
        org_employee_count = {}
        for emp in self.employees_data:
            org_id = emp.get('organization_id')
            if org_id:
                org_employee_count[org_id] = org_employee_count.get(org_id, 0) + 1
        
        if not org_employee_count:
            return "❌ No employees found in any organization."
        
        # Find organization with least employees
        min_org_id = min(org_employee_count, key=org_employee_count.get)
        min_count = org_employee_count[min_org_id]
        
        # Get organization name
        org_name = "Unknown"
        for org in self.organizations_data:
            if org.get('id') == min_org_id:
                org_name = org.get('name', 'Unknown')
                break
        
        return f"📉 **Smallest Organization**: **{org_name}** with {min_count} employees"
    
    def _generate_groq_response(self, question: str, context: str) -> str:
        """Generate response using Groq API"""
        if not self.groq_client:
            return "❌ Groq API not configured. Please set GROQ_API_KEY environment variable."
        
        try:
            # Remove hidden reasoning from model outputs
            THINK_TAG_RE = re.compile(r"<think>[\s\S]*?</think>", re.IGNORECASE)
            
            def hide_thinking(text: str) -> str:
                if not isinstance(text, str):
                    try:
                        text = str(text)
                    except Exception:
                        return ""
                text = THINK_TAG_RE.sub("", text)
                drop_prefixes = ("reasoning:", "thought:", "chain-of-thought:", "scratchpad:")
                lines = [ln for ln in text.splitlines() if not ln.strip().lower().startswith(drop_prefixes)]
                return "\n".join(lines).strip()
            
            prompt = f"""You are an AI HR Assistant. Answer the question using only the context provided below. 
            Be helpful, professional, and provide accurate information based on the HR data.
            
            Context: {context}
            Question: {question}
            
            Provide a clear, helpful response based on the HR data. If the context doesn't contain enough information, 
            politely explain what information is available and suggest what the user might ask instead.
            """
            
            # Get model from environment variable or use default
            model = os.getenv('GROQ_MODEL', 'openai/gpt-oss-120b')
            temperature = float(os.getenv('GROQ_TEMPERATURE', '0.1'))
            max_tokens = int(os.getenv('GROQ_MAX_TOKENS', '1000'))
            
            response = self.groq_client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            answer = response.choices[0].message.content
            return hide_thinking(answer)
            
        except Exception as e:
            return f"❌ Error generating response: {str(e)}"
    
    def _handle_greeting(self) -> str:
        """Handle greeting messages"""
        return "👋 Hello! My name is AI HR Assistant. How may I help you today? I can assist you with:\n\n• Employee information and analytics\n• Organization data and insights\n• HR queries and reports\n• General questions about your workforce\n\nWhat would you like to know?"
    
    def _handle_capabilities_question(self) -> str:
        """Handle questions about capabilities"""
        return "🤖 **AI HR Assistant Capabilities**:\n\n**📊 Data Analytics:**\n• Organizations ranked by employee count\n• Employee breakdown by organization\n• Role analysis and distribution\n• Largest/smallest organization identification\n\n**🔍 Information Retrieval:**\n• Employee search and details\n• Organization information\n• Contact information lookup\n• Role-based queries\n\n**💬 Conversational:**\n• General HR questions\n• Workforce insights\n• Data analysis and reporting\n\nHow can I assist you today?"
    
    def _handle_help_question(self) -> str:
        """Handle help requests"""
        return "🆘 **How I Can Help You:**\n\n**Try asking me:**\n• \"List organizations with most employees\"\n• \"Show employee breakdown by organization\"\n• \"What roles are available?\"\n• \"Which is the largest organization?\"\n• \"How many employees do we have?\"\n• \"Show me all organizations\"\n\n**I can also help with:**\n• General HR questions\n• Workforce analytics\n• Employee information\n• Organization insights\n\nWhat specific information do you need?"
    
    def _handle_thanks(self) -> str:
        """Handle thank you messages"""
        return "😊 You're very welcome! I'm here to help with all your HR needs. Feel free to ask me anything about your workforce, organizations, or employees. Is there anything else I can assist you with?"
    
    def _is_general_question(self, question_lower: str) -> bool:
        """Check if this is a general non-HR question"""
        general_keywords = [
            'time', 'date', 'weather', 'news', 'what is', 'what are', 'explain',
            'tell me about', 'how to', 'why', 'when', 'where', 'who is',
            'mathematics', 'science', 'history', 'geography', 'politics',
            'sports', 'entertainment', 'food', 'travel', 'shopping'
        ]
        
        # Check if question contains general keywords and no HR keywords
        hr_keywords = ['employee', 'organization', 'company', 'hr', 'staff', 'workforce', 'role', 'position']
        
        has_general_keywords = any(keyword in question_lower for keyword in general_keywords)
        has_hr_keywords = any(keyword in question_lower for keyword in hr_keywords)
        
        return has_general_keywords and not has_hr_keywords
    
    def _handle_general_question(self, question: str) -> str:
        """Handle general questions that don't match HR data"""
        question_lower = question.lower()
        
        # Handle common general questions
        if any(phrase in question_lower for phrase in ['what is', 'what are', 'explain', 'tell me about']):
            return "🤔 I'd be happy to help explain things! However, I'm specifically designed to assist with HR-related questions about your workforce, employees, and organizations. Could you rephrase your question to be more specific about HR data, or ask me something like:\n\n• 'What roles are available in our organization?'\n• 'How many employees do we have?'\n• 'Show me our organizations'"
        
        elif any(phrase in question_lower for phrase in ['who are you', 'what do you do', 'introduce yourself']):
            return "🤖 I'm your AI HR Assistant! I specialize in helping you with:\n\n• **Employee Management**: Information about your workforce\n• **Organization Analytics**: Insights about your companies\n• **HR Reports**: Data analysis and reporting\n• **Workforce Insights**: Employee statistics and trends\n\nI can analyze your HR data to provide detailed insights, rankings, and reports. What would you like to know about your workforce?"
        
        elif any(phrase in question_lower for phrase in ['weather', 'time', 'date', 'news']):
            return "🌤️ I'm an HR Assistant focused on workforce data and analytics. I don't have access to weather, time, or news information. But I can help you with:\n\n• Employee information and analytics\n• Organization data and insights\n• HR queries and workforce reports\n\nIs there anything about your HR data I can help you with?"
        
        else:
            return "🤖 I'm your AI HR Assistant, specialized in workforce analytics and HR data. I can help you with:\n\n• Employee information and statistics\n• Organization analytics and rankings\n• HR reports and insights\n• Workforce data analysis\n\nCould you ask me something about your employees, organizations, or HR data? For example:\n• 'How many employees do we have?'\n• 'List organizations with most employees'\n• 'What roles are available?'"

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
        st.header("📊 Data Status")
        st.info(f"Organizations: {len(hr_assistant.organizations_data)}")
        st.info(f"Employees: {len(hr_assistant.employees_data)}")
        
        if len(hr_assistant.organizations_data) > 0 and len(hr_assistant.employees_data) > 0:
            st.success("✅ Data automatically loaded!")
        else:
            st.warning("⚠️ No data available")
        
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
            "List organizations with most employees",
            "Show employee breakdown by organization",
            "What roles are available?",
            "Show me all organizations",
            "Which is the largest organization?",
            "Which is the smallest organization?",
            "Show roles by organization"
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
