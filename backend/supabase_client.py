# supabase_client.py - Supabase Database Client Configuration
import os
import json
import logging
from datetime import datetime
from typing import Optional, Dict, Any, List
from dotenv import load_dotenv
from supabase import create_client, Client

# Load environment variables
load_dotenv()

# Environment variables
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("SupabaseClient")

class SupabaseClient:
    """Supabase client for handling chat storage and client data."""
    
    def __init__(self):
        """Initialize Supabase client."""
        if not SUPABASE_URL or not SUPABASE_KEY:
            raise ValueError("SUPABASE_URL and SUPABASE_KEY must be set in environment variables")
        
        # Use service role key for bypassing RLS if needed
        self.client: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
        logger.info("Supabase client initialized successfully")
    
    async def save_chat_history(self, name: str, company_name: str, chat_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Save chat history to Supabase chat_history table.
        
        Args:
            name: Client name
            company_name: Company name
            chat_history: List of chat messages with timestamp, speaker, and message
            
        Returns:
            Dict containing the saved chat record
        """
        try:
            # Prepare the chat data
            chat_data = {
                "name": name,
                "company": company_name,  # Fixed: using 'company' instead of 'company_name'
                "chat_history": json.dumps(chat_history)  # Store as JSON string
            }
            
            # Insert into chat_history table
            try:
                # First try normal insert
                result = self.client.table("chat_history").insert(chat_data).execute()
            except Exception as insert_error:
                error_msg = str(insert_error).lower()
                if 'row-level security' in error_msg or '42501' in error_msg:
                    logger.error("RLS policy is blocking the insert. Please disable RLS on chat_history table or create appropriate policies.")
                    logger.error("To fix: Go to Supabase Dashboard > Authentication > Policies > chat_history table")
                    logger.error("Either disable RLS or create a policy that allows INSERT operations")
                    return {"error": "Database security policy blocks insert. Please check Supabase RLS settings."}
                else:
                    logger.error(f"Database insert failed: {insert_error}")
                    return {"error": str(insert_error)}
            
            if result.data:
                logger.info(f"Successfully saved chat for {name} from {company_name} with {len(chat_history)} messages")
                return result.data[0]
            else:
                logger.error(f"Failed to save chat history: {result}")
                return {"error": "Failed to save chat history"}
                
        except Exception as e:
            logger.error(f"Error saving chat history: {str(e)}")
            return {"error": str(e)}
    
    async def search_client_by_company(self, company_name: str) -> Optional[Dict[str, Any]]:
        """
        Search for client information by company name.
        
        Args:
            company_name: Company name to search for
            
        Returns:
            Client data if found, None otherwise
        """
        try:
            # Search in clients table (you may need to adjust table name)
            result = self.client.table("clients").select("*").ilike("company", f"%{company_name}%").execute()
            
            if result.data and len(result.data) > 0:
                logger.info(f"Found client data for company: {company_name}")
                return result.data[0]
            else:
                logger.info(f"No client data found for company: {company_name}")
                return None
                
        except Exception as e:
            logger.error(f"Error searching for client: {str(e)}")
            return None
    
    async def search_client_by_name(self, name: str) -> Optional[Dict[str, Any]]:
        """
        Search for client information by name.
        
        Args:
            name: Client name to search for
            
        Returns:
            Client data if found, None otherwise
        """
        try:
            # Search in clients table
            result = self.client.table("clients").select("*").ilike("name", f"%{name}%").execute()
            
            if result.data and len(result.data) > 0:
                logger.info(f"Found client data for name: {name}")
                return result.data[0]
            else:
                logger.info(f"No client data found for name: {name}")
                return None
                
        except Exception as e:
            logger.error(f"Error searching for client: {str(e)}")
            return None
    
    async def get_chat_history(self, name: Optional[str] = None, company_name: Optional[str] = None, limit: int = 50) -> List[Dict[str, Any]]:
        """
        Retrieve chat history from Supabase.
        
        Args:
            name: Optional name filter
            company_name: Optional company name filter
            limit: Maximum number of records to return
            
        Returns:
            List of chat history records
        """
        try:
            query = self.client.table("chat_history").select("*").order("created_at", desc=True).limit(limit)
            
            if name:
                query = query.ilike("name", f"%{name}%")
            if company_name:
                query = query.ilike("company_name", f"%{company_name}%")
            
            result = query.execute()
            
            if result.data:
                logger.info(f"Retrieved {len(result.data)} chat history records")
                return result.data
            else:
                logger.info("No chat history found")
                return []
                
        except Exception as e:
            logger.error(f"Error retrieving chat history: {str(e)}")
            return []
    
    def test_connection(self) -> bool:
        """
        Test the Supabase connection.
        
        Returns:
            True if connection is successful, False otherwise
        """
        try:
            # Try to query the chat_history table to test connection
            result = self.client.table("chat_history").select("count", count="exact").limit(1).execute()
            logger.info("Supabase connection test successful")
            return True
        except Exception as e:
            logger.error(f"Supabase connection test failed: {str(e)}")
            return False

# Global instance
supabase_client = None

def get_supabase_client() -> SupabaseClient:
    """Get or create Supabase client instance."""
    global supabase_client
    if supabase_client is None:
        supabase_client = SupabaseClient()
    return supabase_client

def format_chat_message(timestamp: str, speaker: str, message: str, message_type: str = "text") -> Dict[str, Any]:
    """
    Format a chat message for storage.
    
    Args:
        timestamp: Message timestamp
        speaker: Speaker name (User, Agent, System)
        message: Message content
        message_type: Type of message (text, system, etc.)
        
    Returns:
        Formatted message dictionary
    """
    return {
        "timestamp": timestamp,
        "speaker": speaker,
        "message": message,
        "type": message_type
    }

if __name__ == "__main__":
    # Test the client
    client = get_supabase_client()
    success = client.test_connection()
    print(f"Supabase connection: {'✅ Success' if success else '❌ Failed'}")