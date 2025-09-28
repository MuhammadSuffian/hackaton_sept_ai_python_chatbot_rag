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
    page_title=" Ask HR",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": None,
        "Report a bug": None,
        "About": None
    }
)

# # Chat UI styles
st.markdown(
    """
    <style>
    
      .chat-messages { display:flex; flex-direction:column; gap:18px; flex: 1; overflow-y: auto; padding-right: 10px; }
      .chat-bubble { max-width:62%; padding:12px 16px; border-radius:18px; font-size:16px; line-height:1.45; }
      .chat-bot { background:#e7ddfb; color:#1c1433; align-self:flex-start; }
      .chat-user { background:#d5f7f7; color:#103b3b; align-self:flex-end; }
      .chat-icon-btn { width:40px; height:40px; border:1px solid #e5e7eb; border-radius:10px; background:#ffffff; cursor:pointer; font-size:18px; display:flex; align-items:center; justify-content:center; }
      .chat-send-btn { width:44px; height:44px; border:1px solid #e5e7eb; border-radius:10px; background:#1e66f5; color:#fff; font-weight:600; cursor:pointer; display:flex; align-items:center; justify-content:center; }
      .chat-text { flex:1; border:1px solid #e5e7eb; border-radius:10px; padding:10px 14px; background:#f3f5f9; }
      /* Force consistent widget heights in input row */
      div.stTextInput > label { display:none; }
      div.stTextInput > div { align-items:center; }
      div.stTextInput > div > div input { height:44px !important; line-height:24px !important; padding-top:8px !important; padding-bottom:8px !important; }
      div.stButton > button { height:44px !important; width:44px !important; border-radius:10px !important; display:flex; align-items:center; justify-content:center; padding:0 !important; margin-top:0 !important; }
      /* Fix title clipping with proper spacing */

      h1 { margin-top: 0 !important; padding-top: 0 !important; margin-bottom: 0.5rem !important; }
      /* Ensure title has enough space */
      .main .block-container { padding-top: 2rem !important; }

      /* Hide Streamlit default toolbar/menu/deploy */
      [data-testid="stToolbar"] { display: none !important; }
      #MainMenu { visibility: hidden; }
      footer { visibility: hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)

# Voice input removed per user request

class AIHRAssistant:
    def __init__(self):
        """Initialize  Ask HR with Supabase connection and RAG capabilities"""
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
        self.leaves_data = []
        self.attendance_data = []
        self.embeddings = None
        self.text_chunks = []
        self.vectorstore = None
        
        # UI status messages control
        self.show_status = False
        
        # Load embedding model
        self._load_embedding_model()
    
    def _load_embedding_model(self):
        """Load the sentence transformer model for embeddings"""
        try:
            with st.spinner("Loading AI model..."):
                self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            if self.show_status:
                st.success("AI model loaded successfully!")
            
            # Automatically load data after model loads
            self._auto_load_data()
        except Exception as e:
            st.error(f" Error loading AI model: {str(e)}")
    
    def _auto_load_data(self):
        """Automatically load all database data after AI model loads"""
        try:
            with st.spinner("Loading database data..."):
                # Fetch all tables
                organizations = self.fetch_organizations()
                employees = self.fetch_employees()
                leaves = self.fetch_leaves()
                attendance = self.fetch_attendance()
                
                # Store attendance data
                self.attendance_data = attendance
                self.leaves_data = leaves
                
                if organizations and employees:
                    # Prepare RAG data
                    self.prepare_rag_data()
                    if self.show_status:
                        st.success("✅ All data loaded successfully!")
                else:
                    st.warning("⚠️ No data found in database")
                
                # Debug attendance loading
                if attendance and self.show_status:
                    st.info(f"📊 Loaded {len(attendance)} attendance records")
                else:
                    if self.show_status:
                        st.warning("⚠️ No attendance records found")
        except Exception as e:
            st.error(f"  Error loading data: {str(e)}")
    
    def fetch_organizations(self) -> List[Dict[Any, Any]]:
        """Fetch all organizations data"""
        try:
            response = self.supabase.table('organizations').select("*").execute()
            self.organizations_data = response.data
            return response.data
        except Exception as e:
            st.error(f"  Error fetching organizations: {str(e)}")
            return []
    
    def fetch_employees(self) -> List[Dict[Any, Any]]:
        """Fetch all employees data"""
        try:
            response = self.supabase.table('employees').select("*").execute()
            self.employees_data = response.data
            return response.data
        except Exception as e:
            st.error(f"  Error fetching employees: {str(e)}")
            return []
    
    def fetch_employee_by_organization(self, org_id: int) -> List[Dict[Any, Any]]:
        """Fetch employees by organization ID"""
        try:
            response = self.supabase.table('employees').select("*").eq('organization_id', org_id).execute()
            return response.data
        except Exception as e:
            st.error(f"  Error fetching employees by organization: {str(e)}")
            return []
    
    def fetch_leaves(self) -> List[Dict[Any, Any]]:
        """Fetch all leaves data"""
        try:
            response = self.supabase.table('leaves').select("*").execute()
            return response.data
        except Exception as e:
            st.error(f"  Error fetching leaves: {str(e)}")
            return []
    
    def fetch_attendance(self) -> List[Dict[Any, Any]]:
        """Fetch all attendance data"""
        try:
            response = self.supabase.table('attendance').select("*").execute()
            return response.data
        except Exception as e:
            st.error(f"  Error fetching attendance: {str(e)}")
            return []
    
    def prepare_rag_data(self):
        """Prepare data for RAG system by creating text chunks and embeddings"""
        if not self.embedding_model:
            st.error("  AI model not loaded!")
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
            
            chunk = f"Employee: {emp.get('name', 'Unknown')} - Email: {emp.get('email', 'Unknown')} - Role: {emp.get('role', 'Unknown')} - Phone: {emp.get('phone', 'Unknown')} - Department: {emp.get('department', 'Unknown')} - Salary: {emp.get('salary', 'Unknown')} - Leave Balance: {emp.get('leave_balance', 'Unknown')} - Status: {emp.get('status', 'Unknown')} - Organization: {org_name}"
            self.text_chunks.append(chunk)
            documents.append(Document(page_content=chunk, metadata={"type": "employee", "id": emp.get('id')}))
        
        # Create text chunks from leaves
        for leave in self.leaves_data:
            # Find employee name
            emp_name = "Unknown"
            for emp in self.employees_data:
                if emp.get('id') == leave.get('employee_id'):
                    emp_name = emp.get('name', 'Unknown')
                    break
            
            chunk = f"Leave: Employee {emp_name} - Type: {leave.get('type', 'Unknown')} - Start: {leave.get('start_date', 'Unknown')} - End: {leave.get('end_date', 'Unknown')} - Reason: {leave.get('reason', 'Unknown')} - Status: {leave.get('status', 'Unknown')}"
            self.text_chunks.append(chunk)
            documents.append(Document(page_content=chunk, metadata={"type": "leave", "id": leave.get('id')}))
        
        # Create text chunks from attendance
        for att in self.attendance_data:
            # Find employee name
            emp_name = "Unknown"
            for emp in self.employees_data:
                if emp.get('id') == att.get('employee_id'):
                    emp_name = emp.get('name', 'Unknown')
                    break
            
            chunk = f"Attendance: Employee {emp_name} - Date: {att.get('work_date', 'Unknown')} - Check In: {att.get('check_in', 'Unknown')} - Check Out: {att.get('check_out', 'Unknown')} - Status: {att.get('status', 'Unknown')}"
            self.text_chunks.append(chunk)
            documents.append(Document(page_content=chunk, metadata={"type": "attendance", "id": att.get('id')}))
        
        # Create FAISS vectorstore
        if documents:
            with st.spinner("Creating vectorstore..."):
                embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
                self.vectorstore = FAISS.from_documents(documents, embeddings)
            if self.show_status:
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
            st.error(f"  Error in similarity search: {str(e)}")
            return []
    
    def answer_hr_question(self, question: str) -> str:
        """Answer HR questions using RAG with enhanced analytical capabilities"""
        question_lower = question.lower().strip()
        
        
        # Check if data is loaded for HR-specific queries first
        if not self.organizations_data and not self.employees_data:
            return "  No HR data loaded. Please wait for automatic data loading to complete. I can still help with general questions though!"
        
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
        
        # New analytical queries for leaves and attendance
        elif any(phrase in question_lower for phrase in ['leave analytics', 'leave statistics', 'leave data', 'leaves summary', 'leave report']):
            return self._get_leave_analytics()
        
        elif any(phrase in question_lower for phrase in ['attendance analytics', 'attendance statistics', 'attendance data', 'attendance summary', 'attendance report']):
            return self._get_attendance_analytics()
        
        elif any(phrase in question_lower for phrase in ['detailed attendance', 'comprehensive attendance', 'attendance breakdown', 'attendance analysis']):
            return self._get_detailed_attendance_analytics()
        
        elif any(phrase in question_lower for phrase in ['list attendance', 'show attendance', 'attendance list', 'all attendance', 'attendance records', 'attendance data']):
            return self._get_all_attendance()
        
        elif any(phrase in question_lower for phrase in ['employee attendance', 'attendance for', 'show attendance for', 'attendance for', 'show attendence for']):
            # Extract employee name from query
            employee_name = self._extract_employee_name(question)
            if employee_name:
                return self._get_employee_attendance(employee_name)
            else:
                return "  Please specify the employee name. Example: 'Show attendance for John Smith'"
        
        elif any(phrase in question_lower for phrase in ['employee details', 'employee info', 'employee information', 'show employee']):
            # Extract employee name from query
            employee_name = self._extract_employee_name(question)
            if employee_name:
                return self._get_employee_details(employee_name)
            else:
                return "  Please specify the employee name. Example: 'Show details for John Smith'"
        
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
            st.error(f"  Error with Groq API: {str(e)}")
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
            return f"🤖 ** Ask HR Response**:\n\nBased on the available data, here's what I found:\n\n{context}\n\n*Relevance scores: {[f'{chunk[1]:.2f}' for chunk in relevant_chunks]}*"
    
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
            return "  No data available for analysis."
        
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
            return "  No data available for analysis."
        
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
            return "  No data available for analysis."
        
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
            return "  No data available for analysis."
        
        # Count employees per organization
        org_employee_count = {}
        for emp in self.employees_data:
            org_id = emp.get('organization_id')
            if org_id:
                org_employee_count[org_id] = org_employee_count.get(org_id, 0) + 1
        
        if not org_employee_count:
            return "  No employees found in any organization."
        
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
            return "  No data available for analysis."
        
        # Count employees per organization
        org_employee_count = {}
        for emp in self.employees_data:
            org_id = emp.get('organization_id')
            if org_id:
                org_employee_count[org_id] = org_employee_count.get(org_id, 0) + 1
        
        if not org_employee_count:
            return "  No employees found in any organization."
        
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
    
    def _get_leave_analytics(self) -> str:
        """Get leave analytics and statistics"""
        if not self.leaves_data:
            return "  No leave data available."
        
        # Count leaves by type
        leave_types = {}
        leave_status = {}
        for leave in self.leaves_data:
            leave_type = leave.get('type', 'Unknown')
            status = leave.get('status', 'Unknown')
            
            leave_types[leave_type] = leave_types.get(leave_type, 0) + 1
            leave_status[status] = leave_status.get(status, 0) + 1
        
        result = "📅 **Leave Analytics**:\n\n"
        result += f"**Total Leaves**: {len(self.leaves_data)}\n\n"
        
        result += "**By Type**:\n"
        for leave_type, count in sorted(leave_types.items(), key=lambda x: x[1], reverse=True):
            result += f"• {leave_type}: {count}\n"
        
        result += "\n**By Status**:\n"
        for status, count in sorted(leave_status.items(), key=lambda x: x[1], reverse=True):
            result += f"• {status}: {count}\n"
        
        return result
    
    def _get_attendance_analytics(self) -> str:
        """Get attendance analytics and statistics"""
        if not self.attendance_data:
            return "  No attendance data available."
        
        # Count attendance by status
        attendance_status = {}
        # Track check-in/out patterns
        check_in_times = []
        check_out_times = []
        # Track work dates
        work_dates = set()
        
        for att in self.attendance_data:
            status = att.get('status', 'Unknown')
            attendance_status[status] = attendance_status.get(status, 0) + 1
            
            # Collect check-in/out times for analysis
            check_in = att.get('check_in')
            check_out = att.get('check_out')
            work_date = att.get('work_date')
            
            if check_in:
                check_in_times.append(check_in)
            if check_out:
                check_out_times.append(check_out)
            if work_date:
                work_dates.add(work_date)
        
        result = "⏰ **Attendance Analytics**:\n\n"
        result += f"**Total Records**: {len(self.attendance_data)}\n"
        result += f"**Unique Work Dates**: {len(work_dates)}\n\n"
        
        result += "**By Status**:\n"
        for status, count in sorted(attendance_status.items(), key=lambda x: x[1], reverse=True):
            result += f"• {status}: {count}\n"
        
        # Add check-in/out analysis if data available
        if check_in_times:
            result += f"\n**Check-in Records**: {len(check_in_times)}\n"
        if check_out_times:
            result += f"**Check-out Records**: {len(check_out_times)}\n"
        
        return result
    
    def _get_detailed_attendance_analytics(self) -> str:
        """Get detailed attendance analytics using all attendance fields"""
        if not self.attendance_data:
            return "  No attendance data available."
        
        # Analyze attendance patterns
        attendance_by_date = {}
        attendance_by_employee = {}
        status_counts = {}
        check_in_analysis = []
        check_out_analysis = []
        
        for att in self.attendance_data:
            employee_id = att.get('employee_id')
            work_date = att.get('work_date')
            check_in = att.get('check_in')
            check_out = att.get('check_out')
            status = att.get('status', 'Unknown')
            
            # Count by status
            status_counts[status] = status_counts.get(status, 0) + 1
            
            # Group by date
            if work_date:
                if work_date not in attendance_by_date:
                    attendance_by_date[work_date] = []
                attendance_by_date[work_date].append(att)
            
            # Group by employee
            if employee_id:
                if employee_id not in attendance_by_employee:
                    attendance_by_employee[employee_id] = []
                attendance_by_employee[employee_id].append(att)
            
            # Collect check-in/out data
            if check_in:
                check_in_analysis.append({
                    'employee_id': employee_id,
                    'date': work_date,
                    'time': check_in,
                    'status': status
                })
            if check_out:
                check_out_analysis.append({
                    'employee_id': employee_id,
                    'date': work_date,
                    'time': check_out,
                    'status': status
                })
        
        result = "⏰ **Detailed Attendance Analytics**:\n\n"
        result += f"**Total Attendance Records**: {len(self.attendance_data)}\n"
        result += f"**Unique Work Dates**: {len(attendance_by_date)}\n"
        result += f"**Employees with Attendance**: {len(attendance_by_employee)}\n\n"
        
        result += "**By Status**:\n"
        for status, count in sorted(status_counts.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / len(self.attendance_data)) * 100
            result += f"• {status}: {count} ({percentage:.1f}%)\n"
        
        result += f"\n**Check-in Records**: {len(check_in_analysis)}\n"
        result += f"**Check-out Records**: {len(check_out_analysis)}\n"
        
        # Show recent work dates
        if attendance_by_date:
            recent_dates = sorted(attendance_by_date.keys(), reverse=True)[:5]
            result += f"\n**Recent Work Dates** (last 5):\n"
            for date in recent_dates:
                count = len(attendance_by_date[date])
                result += f"• {date}: {count} records\n"
        
        return result
    
    def _get_all_attendance(self) -> str:
        """Get all attendance records in a formatted list"""
        if not self.attendance_data:
            return "  No attendance data available. The attendance table appears to be empty or not accessible."
        
        if not self.employees_data:
            return "  No employee data available to match attendance records."
        
        # Create a mapping of employee_id to employee_name
        employee_map = {}
        for emp in self.employees_data:
            employee_map[emp.get('id')] = emp.get('name', 'Unknown')
        
        result = f"⏰ **All Attendance Records** ({len(self.attendance_data)} total):\n\n"
        
        # Sort attendance by date (most recent first)
        sorted_attendance = sorted(self.attendance_data, 
                                 key=lambda x: x.get('work_date', ''), 
                                 reverse=True)
        
        # Group by date for better organization
        attendance_by_date = {}
        for att in sorted_attendance:
            work_date = att.get('work_date', 'Unknown')
            if work_date not in attendance_by_date:
                attendance_by_date[work_date] = []
            attendance_by_date[work_date].append(att)
        
        # Display attendance grouped by date
        for date in sorted(attendance_by_date.keys(), reverse=True)[:10]:  # Show last 10 days
            result += f"📅 **{date}** ({len(attendance_by_date[date])} records):\n"
            for att in attendance_by_date[date]:
                employee_id = att.get('employee_id')
                employee_name = employee_map.get(employee_id, f"Employee ID: {employee_id}")
                check_in = att.get('check_in', 'N/A')
                check_out = att.get('check_out', 'N/A')
                status = att.get('status', 'Unknown')
                
                result += f"  • **{employee_name}**: {check_in} - {check_out} ({status})\n"
            result += "\n"
        
        if len(attendance_by_date) > 10:
            result += f"... and {len(attendance_by_date) - 10} more days of records.\n"
        
        # Add summary statistics
        status_counts = {}
        for att in self.attendance_data:
            status = att.get('status', 'Unknown')
            status_counts[status] = status_counts.get(status, 0) + 1
        
        result += "\n**Summary by Status**:\n"
        for status, count in sorted(status_counts.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / len(self.attendance_data)) * 100
            result += f"• {status}: {count} ({percentage:.1f}%)\n"
        
        return result
    
    def _get_employee_attendance(self, employee_name: str) -> str:
        """Get attendance details for a specific employee"""
        if not self.attendance_data or not self.employees_data:
            return "  No attendance or employee data available."
        
        # Find employee by name (exact match first, then partial match)
        employee = None
        for emp in self.employees_data:
            emp_name = emp.get('name', '').lower()
            search_name = employee_name.lower()
            
            # Exact match
            if emp_name == search_name:
                employee = emp
                break
            # Partial match (first name or last name)
            elif search_name in emp_name or any(part in emp_name for part in search_name.split()):
                employee = emp
                break
        
        if not employee:
            return f"  Employee '{employee_name}' not found. Available employees: {', '.join([emp.get('name', 'Unknown') for emp in self.employees_data[:5]])}..."
        
        employee_id = employee.get('id')
        employee_attendance = [att for att in self.attendance_data if att.get('employee_id') == employee_id]
        
        if not employee_attendance:
            return f"  No attendance records found for {employee_name}."
        
        # Analyze employee's attendance
        status_counts = {}
        recent_attendance = []
        
        for att in employee_attendance:
            status = att.get('status', 'Unknown')
            status_counts[status] = status_counts.get(status, 0) + 1
            
            recent_attendance.append({
                'date': att.get('work_date', 'Unknown'),
                'check_in': att.get('check_in', 'Unknown'),
                'check_out': att.get('check_out', 'Unknown'),
                'status': status
            })
        
        # Sort by date (most recent first)
        recent_attendance.sort(key=lambda x: x['date'], reverse=True)
        
        result = f"⏰ **Attendance for {employee_name}**:\n\n"
        result += f"**Total Records**: {len(employee_attendance)}\n\n"
        
        result += "**Status Summary**:\n"
        for status, count in sorted(status_counts.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / len(employee_attendance)) * 100
            result += f"• {status}: {count} ({percentage:.1f}%)\n"
        
        result += f"\n**Recent Attendance** (last 5 records):\n"
        for att in recent_attendance[:5]:
            result += f"• {att['date']}: {att['check_in']} - {att['check_out']} ({att['status']})\n"
        
        return result
    
    def _get_employee_details(self, employee_name: str) -> str:
        """Get detailed information about a specific employee"""
        if not self.employees_data:
            return "  No employee data available."
        
        # Find employee by name
        employee = None
        for emp in self.employees_data:
            if emp.get('name', '').lower() == employee_name.lower():
                employee = emp
                break
        
        if not employee:
            return f"  Employee '{employee_name}' not found."
        
        # Get organization name
        org_name = "Unknown"
        for org in self.organizations_data:
            if org.get('id') == employee.get('organization_id'):
                org_name = org.get('name', 'Unknown')
                break
        
        result = f"👤 **Employee Details - {employee.get('name', 'Unknown')}**:\n\n"
        result += f"**Basic Info**:\n"
        result += f"• Email: {employee.get('email', 'Unknown')}\n"
        result += f"• Phone: {employee.get('phone', 'Unknown')}\n"
        result += f"• Role: {employee.get('role', 'Unknown')}\n"
        result += f"• Department: {employee.get('department', 'Unknown')}\n"
        result += f"• Organization: {org_name}\n"
        result += f"• Status: {employee.get('status', 'Unknown')}\n"
        result += f"• Salary: {employee.get('salary', 'Unknown')}\n"
        result += f"• Leave Balance: {employee.get('leave_balance', 'Unknown')}\n"
        result += f"• Hire Date: {employee.get('hire_date', 'Unknown')}\n"
        result += f"• Last Login: {employee.get('last_login', 'Unknown')}\n"
        
        return result
    
    def _extract_employee_name(self, question: str) -> str:
        """Extract employee name from query"""
        import re
        
        # Pattern to match "for [Name]" or just "[Name]"
        patterns = [
            r'for\s+([A-Za-z\s]+)',
            r'([A-Z][a-z]+\s+[A-Z][a-z]+)',  # First Last
            r'([A-Z][a-z]+)',  # Single name
            r'employee\s+([A-Za-z\s]+)',
            r'details\s+([A-Za-z\s]+)',
            r'attendance\s+for\s+([A-Za-z\s]+)',
            r'show\s+attendance\s+for\s+([A-Za-z\s]+)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, question, re.IGNORECASE)
            if match:
                name = match.group(1).strip()
                # Remove common words
                name = re.sub(r'\b(for|employee|details|info|information|attendance|show)\b', '', name, flags=re.IGNORECASE).strip()
                if name:
                    return name
        
        return None
    
    def _generate_groq_response(self, question: str, context: str) -> str:
        """Generate response using Groq API"""
        if not self.groq_client:
            return "  Groq API not configured. Please set GROQ_API_KEY environment variable."
        
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
            
            prompt = f"""You are an  Ask HR. Answer the question using only the context provided below. 
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
            return f"  Error generating response: {str(e)}"
    
    def _handle_greeting(self) -> str:
        """Handle greeting messages"""
        return "👋 Hello! My name is  Ask HR. How may I help you today? I can assist you with:\n\n• Employee information and analytics\n• Organization data and insights\n• HR queries and reports\n• General questions about your workforce\n\nWhat would you like to know?"
    
    def _handle_capabilities_question(self) -> str:
        """Handle questions about capabilities"""
        return "🤖 ** Ask HR Capabilities**:\n\n**📊 Data Analytics:**\n• Organizations ranked by employee count\n• Employee breakdown by organization\n• Role analysis and distribution\n• Largest/smallest organization identification\n\n**🔍 Information Retrieval:**\n• Employee search and details\n• Organization information\n• Contact information lookup\n• Role-based queries\n\n**💬 Conversational:**\n• General HR questions\n• Workforce insights\n• Data analysis and reporting\n\nHow can I assist you today?"
    
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
            return "🤖 I'm your  Ask HR! I specialize in helping you with:\n\n• **Employee Management**: Information about your workforce\n• **Organization Analytics**: Insights about your companies\n• **HR Reports**: Data analysis and reporting\n• **Workforce Insights**: Employee statistics and trends\n\nI can analyze your HR data to provide detailed insights, rankings, and reports. What would you like to know about your workforce?"
        
        elif any(phrase in question_lower for phrase in ['weather', 'time', 'date', 'news']):
            return "🌤️ I'm an HR Assistant focused on workforce data and analytics. I don't have access to weather, time, or news information. But I can help you with:\n\n• Employee information and analytics\n• Organization data and insights\n• HR queries and workforce reports\n\nIs there anything about your HR data I can help you with?"
        
        else:
            return "🤖 I'm your  Ask HR, specialized in workforce analytics and HR data. I can help you with:\n\n• Employee information and statistics\n• Organization analytics and rankings\n• HR reports and insights\n• Workforce data analysis\n\nCould you ask me something about your employees, organizations, or HR data? For example:\n• 'How many employees do we have?'\n• 'List organizations with most employees'\n• 'What roles are available?'"

def main():
    """Main Streamlit app"""
    st.title("🤖  Ask HR")
    st.markdown("---")
    
    # Initialize the  Ask HR
    if 'hr_assistant' not in st.session_state:
        st.session_state.hr_assistant = AIHRAssistant()
    
    hr_assistant = st.session_state.hr_assistant
    
    # Sidebar
    with st.sidebar:
        st.header("Data Status")
        st.info(f"Organizations: {len(hr_assistant.organizations_data)}")
        st.info(f"Employees: {len(hr_assistant.employees_data)}")
        st.info(f"Attendance Records: {len(hr_assistant.attendance_data)}")
        st.info(f"Leave Records: {len(hr_assistant.leaves_data)}")
        
        if len(hr_assistant.organizations_data) > 0 and len(hr_assistant.employees_data) > 0:
            st.success("✅ Data automatically loaded!")
        else:
            st.warning("⚠️ No data available")
        
        # View Data controls removed per request
    
    # Main content area: chat (left) + FAQ (right)
    col_chat, col_faq = st.columns([3, 1])
    
    with col_chat:
        st.header("💬 Chat with  Ask HR")
        
        if "messages" not in st.session_state:
            st.session_state.messages = []
        
        # Chat container wrapper
        st.markdown('<div class="chat-wrapper">', unsafe_allow_html=True)
        
        # Messages container
        st.markdown('<div class="chat-messages">', unsafe_allow_html=True)
        for message in st.session_state.messages:
            if message["role"] == "assistant":
                st.markdown(f'<div class="chat-bubble chat-bot">{message["content"]}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="chat-bubble chat-user">{message["content"]}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Input container
        st.markdown('<div class="chat-input-container">', unsafe_allow_html=True)
        with st.container():
            # Use balanced column widths for alignment
            col_mic, col_input, col_send = st.columns([0.06, 0.88, 0.06])
            with col_mic:
                st.empty()  # voice input removed
            with col_input:
                user_text = st.text_input("", placeholder="Type a message...", key="chat_text")
            with col_send:
                send_clicked = st.button("➤", key="chat_send")

            # Voice input removed per user request
            if send_clicked and user_text:
                st.session_state.messages.append({"role": "user", "content": user_text})
                with st.spinner("Thinking..."):
                    response = hr_assistant.answer_hr_question(user_text)
                st.session_state.messages.append({"role": "assistant", "content": response})
                st.rerun()
        st.markdown('</div></div>', unsafe_allow_html=True)

    with col_faq:
        st.markdown("**FAQ**")
        st.write("How many employees do we have?")
        st.write("List organizations with most employees?")
        st.markdown("**Suggestions**")
        st.write("Check attendance record")
        st.write("View company policies")
        # Quick Actions removed per request

if __name__ == "__main__":
    main()
