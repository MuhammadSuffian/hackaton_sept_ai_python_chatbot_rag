import os
from supabase import create_client, Client
from typing import List, Dict, Any
import json
import streamlit as st
import pandas as pd

# Configure Streamlit page
st.set_page_config(
    page_title="Supabase Data Fetcher",
    page_icon="🗄️",
    layout="wide",
    initial_sidebar_state="expanded"
)
st.title("Supabase Data Fetcher")

class SupabaseDataFetcher:
    def __init__(self):
        """Initialize Supabase client with environment variables"""
        # You need to set these environment variables or replace with your actual values
        self.url = os.getenv('SUPABASE_URL', 'https://ptkqgiqqefoceswfwent.supabase.co')
        self.key = os.getenv('SUPABASE_ANON_KEY', 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InB0a3FnaXFxZWZvY2Vzd2Z3ZW50Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTg5Njk1NDcsImV4cCI6MjA3NDU0NTU0N30.6DU3Ahplr8tybCxBcJScd8PaPqYWXNb_Y7zSMvF3wRg')
        
        if self.url == 'your-supabase-url' or self.key == 'your-supabase-anon-key':
            st.error("⚠️ Please set your Supabase credentials!")
            st.info("Set environment variables or replace the values directly in the code.")
        
        self.supabase: Client = create_client(self.url, self.key)
    
    def fetch_all_data(self, table_name: str) -> List[Dict[Any, Any]]:
        """
        Fetch all data from a specified table
        
        Args:
            table_name (str): Name of the table to fetch data from
            
        Returns:
            List[Dict]: List of records from the table
        """
        try:
            response = self.supabase.table(table_name).select("*").execute()
            return response.data
        except Exception as e:
            st.error(f"❌ Error fetching data from {table_name}: {str(e)}")
            return []
    
    def fetch_data_with_filter(self, table_name: str, column: str, value: Any) -> List[Dict[Any, Any]]:
        """
        Fetch data from a table with a specific filter
        
        Args:
            table_name (str): Name of the table
            column (str): Column to filter by
            value: Value to filter for
            
        Returns:
            List[Dict]: Filtered records
        """
        try:
            response = self.supabase.table(table_name).select("*").eq(column, value).execute()
            return response.data
        except Exception as e:
            st.error(f"❌ Error fetching filtered data: {str(e)}")
            return []
    
    def fetch_limited_data(self, table_name: str, limit: int = 10) -> List[Dict[Any, Any]]:
        """
        Fetch limited number of records from a table
        
        Args:
            table_name (str): Name of the table
            limit (int): Maximum number of records to fetch
            
        Returns:
            List[Dict]: Limited records from the table
        """
        try:
            response = self.supabase.table(table_name).select("*").limit(limit).execute()
            return response.data
        except Exception as e:
            st.error(f"❌ Error fetching limited data: {str(e)}")
            return []

def display_data_streamlit(data: List[Dict[Any, Any]], title: str = "Data from Supabase"):
    """
    Display data in Streamlit interface
    
    Args:
        data (List[Dict]): Data to display
        title (str): Title for the display
    """
    if not data:
        st.warning("❌ No data found or error occurred")
        return
    
    st.subheader(f"📊 {title}")
    st.info(f"📈 Total records: {len(data)}")
    
    # Convert to DataFrame for better display
    df = pd.DataFrame(data)
    
    # Display as table
    st.dataframe(df, use_container_width=True)
    
    # Display raw JSON for detailed view
    with st.expander("🔍 View Raw Data (JSON)"):
        for i, record in enumerate(data, 1):
            st.json(record)

def main():
    """Main Streamlit app"""
    # Header
    st.title("🗄️ Supabase Data Fetcher")
    st.markdown("---")
    
    # Initialize the fetcher
    fetcher = SupabaseDataFetcher()
    
    # Sidebar for controls
    with st.sidebar:
        st.header("⚙️ Controls")
        
        # Table name input
        table_name = st.text_input(
            "📋 Table Name", 
            value="your_table_name",
            help="Enter the name of your Supabase table"
        )
        
        # Fetch options
        st.subheader("🔍 Fetch Options")
        fetch_all = st.checkbox("Fetch All Data", value=True)
        fetch_limited = st.checkbox("Fetch Limited Data", value=True)
        fetch_filtered = st.checkbox("Fetch Filtered Data", value=False)
        
        # Limited data options
        if fetch_limited:
            limit = st.number_input("Limit", min_value=1, max_value=1000, value=10)
        
        # Filter options
        if fetch_filtered:
            st.subheader("🔎 Filter Options")
            filter_column = st.text_input("Filter Column", placeholder="e.g., status")
            filter_value = st.text_input("Filter Value", placeholder="e.g., active")
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("📊 Data Display")
        
        if st.button("🚀 Fetch Data", type="primary"):
            if table_name == "your_table_name":
                st.warning("⚠️ Please enter a valid table name in the sidebar!")
                return
            
            # Fetch all data
            if fetch_all:
                with st.spinner("Fetching all data..."):
                    all_data = fetcher.fetch_all_data(table_name)
                    display_data_streamlit(all_data, f"All data from {table_name}")
            
            # Fetch limited data
            if fetch_limited:
                with st.spinner(f"Fetching limited data (first {limit} records)..."):
                    limited_data = fetcher.fetch_limited_data(table_name, limit)
                    display_data_streamlit(limited_data, f"Limited data from {table_name}")
            
            # Fetch filtered data
            if fetch_filtered and filter_column and filter_value:
                with st.spinner(f"Fetching filtered data..."):
                    filtered_data = fetcher.fetch_data_with_filter(table_name, filter_column, filter_value)
                    display_data_streamlit(filtered_data, f"Filtered data from {table_name}")
    
    with col2:
        st.header("ℹ️ Info")
        st.info("""
        **How to use:**
        1. Enter your table name
        2. Select fetch options
        3. Click 'Fetch Data'
        4. View results below
        """)
        
        st.success("✅ Ready to fetch data!")
        
        # Connection status
        st.subheader("🔗 Connection Status")
        try:
            # Test connection
            test_response = fetcher.supabase.table("_test").select("*").limit(1).execute()
            st.success("🟢 Connected to Supabase")
        except:
            st.error("🔴 Connection failed")

if __name__ == "__main__":
    main()