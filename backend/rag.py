# agent.py – Tekisho Research Assistant (MongoDB + LiveKit Cloud + RAG)
import os
import json
import logging
import re
from dotenv import load_dotenv
from pymongo import MongoClient
from livekit import agents
from livekit.agents import AgentSession, Agent, RoomInputOptions, WorkerOptions
from livekit.agents.llm import function_tool
from livekit.plugins import noise_cancellation, silero, tavus
from prompts import SESSION_INSTRUCTION, AGENT_INSTRUCTION

# RAG module
import rag

# -------------------------------------
# Environment Setup
# -------------------------------------
load_dotenv()

MONGO_URI = os.getenv("MONGO_URI")
DB_NAME = os.getenv("MONGO_DB_NAME", "tekisho_db")
COLLECTION_NAME = os.getenv("MONGO_COLLECTION", "clients")

LIVEKIT_URL = os.getenv("LIVEKIT_URL")
LIVEKIT_API_KEY = os.getenv("LIVEKIT_API_KEY")
LIVEKIT_API_SECRET = os.getenv("LIVEKIT_API_SECRET")

TAVUS_API_KEY = os.getenv("TAVUS_API_KEY")
REPLICA_ID = os.getenv("REPLICA_ID")
PERSONA_ID = os.getenv("PERSONA_ID")

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("TekishoAgent")


# =====================================
# Helper Functions
# =====================================
def format_numbers_for_speech(text: str) -> str:
    """
    Convert numeric patterns to speech-friendly format.
    Examples:
    - "150-200%" -> "one fifty to two hundred percent"
    - "25-35%" -> "twenty five to thirty five percent"
    - "6-8 weeks" -> "six to eight weeks"
    """
    
    def number_to_words(num_str):
        """Convert number string to words (simple version for common cases)"""
        num = int(num_str)
        
        # Handle special cases
        if num == 0:
            return "zero"
        
        ones = ["", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
        teens = ["ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", 
                 "sixteen", "seventeen", "eighteen", "nineteen"]
        tens = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]
        
        if num < 10:
            return ones[num]
        elif num < 20:
            return teens[num - 10]
        elif num < 100:
            return tens[num // 10] + (" " + ones[num % 10] if num % 10 != 0 else "")
        elif num < 1000:
            hundreds = ones[num // 100] + " hundred"
            remainder = num % 100
            if remainder == 0:
                return hundreds
            elif remainder < 10:
                return hundreds + " and " + ones[remainder]
            elif remainder < 20:
                return hundreds + " and " + teens[remainder - 10]
            else:
                return hundreds + " and " + tens[remainder // 10] + (" " + ones[remainder % 10] if remainder % 10 != 0 else "")
        else:
            return num_str  # Fallback for large numbers
    
    # Pattern for percentage ranges like "150-200%"
    def replace_percent_range(match):
        start = match.group(1)
        end = match.group(2)
        start_words = number_to_words(start)
        end_words = number_to_words(end)
        return f"{start_words} to {end_words} percent"
    
    # Pattern for simple ranges like "6-8 weeks"
    def replace_number_range(match):
        start = match.group(1)
        end = match.group(2)
        unit = match.group(3)
        start_words = number_to_words(start)
        end_words = number_to_words(end)
        return f"{start_words} to {end_words} {unit}"
    
    # Pattern for single percentages like "25%"
    def replace_single_percent(match):
        num = match.group(1)
        num_words = number_to_words(num)
        return f"{num_words} percent"
    
    # Apply replacements
    # Range with percentage: 150-200%
    text = re.sub(r'(\d+)-(\d+)%', replace_percent_range, text)
    
    # Range with unit: 6-8 weeks, 25-35 days
    text = re.sub(r'(\d+)-(\d+)\s+(weeks|days|months|hours)', replace_number_range, text)
    
    # Single percentage: 25%
    text = re.sub(r'(\d+)%', replace_single_percent, text)
    
    return text


# =====================================
# Agent Definition
# =====================================
class Assistant(Agent):
    """Tekisho AI Agent with RAG-powered responses and DB retrieval."""

    def _init_(self, instructions=AGENT_INSTRUCTION):
        super()._init_(instructions=instructions)
        self.conversation_context = {
            "client_name": None,
            "company": None,
            "industry": None,
            "mail_id": None,
            "phone_no": None,
            "company_summary": None,
            "research_about_company": None,
            "challenges_discussed": [],
            "greeting_done": False,
            "identity_confirmed": False
        }

    # ------------------
    # Function: Search Client in Database
    # ------------------
    @function_tool()
    async def search_client_in_database(self, name: str, company: str) -> str:
        """
        Search for client information in MongoDB database based on name and company.
        Returns personalized greeting with company research if found.
        """
        try:
            client = MongoClient(MONGO_URI)
            db = client[DB_NAME]
            collection = db[COLLECTION_NAME]
            # Search for client (case-insensitive)
            # Updated query to match new schema structure
            query = {
                "$or": [
                    {"company_name": {"$regex": company, "$options": "i"}},
                    {"company_details.name": {"$regex": name, "$options": "i"}}
                ]
            }
            client_doc = collection.find_one(query)
            if client_doc:
                # Extract data from nested structure
                record_id = str(client_doc.get('_id', ''))
                record_company_name = client_doc.get('company_name', '')
                # Extract from company_details
                company_details = client_doc.get('company_details', {})
                record_name = company_details.get('name', '')
                record_email = company_details.get('email', '')
                record_phone = company_details.get('phone', '')
                # Extract from ai_extracted_data
                ai_data = client_doc.get('ai_extracted_data', {})
                structured_data = ai_data.get('structured_data', {})
                record_research_about_company = structured_data.get('Industry', '')
                record_company_summary = structured_data.get('Description/tagline', '')
                # Store all information in context
                self.conversation_context.update({
                    "record_id": record_id,
                    "client_name": record_name,
                    "company": record_company_name,
                    "mail_id": record_email,
                    "phone_no": record_phone,
                    "company_summary": record_company_summary,
                    "research_about_company": record_research_about_company,
                    "identity_confirmed": True
                })
                # Build personalized response
                response = f"Hi {record_name}! "
                if record_company_name:
                    response += f"I see you're from {record_company_name}. "
                if record_research_about_company:
                    response += f"Your company operates in the {record_research_about_company} industry. "
                if record_company_summary:
                    response += f"{record_company_summary} "
                response += "It's wonderful to connect with you! What specific challenges or opportunities can I help you explore today?"
                logger.info(f"Found client in DB: {record_name} from {record_company_name}")
                # Close the connection
                client.close()
                return response
            else:
                # Not found in database
                logger.info(f"Client not found: {name} from {company}")
                client.close()
                return (f"Nice to meet you, {name}! I don't have prior information about {company} in our system yet, "
                       f"but I'd love to learn more about your business and the challenges you're facing. "
                       f"Could you tell me a bit about what {company} does and what brings you here today?")
        except Exception as e:
            logger.error(f"Database search failed: {e}")
            if 'client' in locals():
                client.close()
            return (f"Great to meet you, {name} from {company}! "
                   f"I'd love to understand more about your business challenges. "
                   f"What specific areas are you looking to improve or automate?")

    # ------------------
    # Function: Get Tekisho Solutions (RAG-powered)
    # ------------------
    @function_tool()
    async def get_tekisho_solutions(
        self, 
        challenge: str, 
        industry: str = None,
        wants_specific_metrics: bool = False
    ) -> str:
        """
        Get AI solutions for a specific business challenge using RAG.
        Returns conversational response with metrics formatted for speech.
        """
        try:
            # Track challenges discussed
            self.conversation_context["challenges_discussed"].append(challenge)
            
            # Use industry from context if not provided
            if not industry and self.conversation_context.get("research_about_company"):
                industry = self.conversation_context["research_about_company"]
            
            # Call RAG function
            answer = rag.get_tekisho_solutions(
                challenge=challenge, 
                industry=industry
            )
            
            # Format numbers for speech
            answer = format_numbers_for_speech(answer)
            
            logger.info(f"Generated solution for challenge: {challenge}")
            return answer
            
        except Exception as e:
            logger.exception("get_tekisho_solutions failed: %s", e)
            fallback = ("Our AI solutions typically deliver ROI ranging from one fifty to three hundred percent "
                       "within the first six to twelve weeks. Cost savings usually fall between twenty five and forty percent, "
                       "with productivity improvements of fifty to eighty percent. "
                       "Would you like me to connect you with a solution architect to discuss specific numbers for your use case?")
            return fallback

    # ------------------
    # Function: Ask Clarifying Question
    # ------------------
    @function_tool()
    async def ask_for_clarification(self, question: str) -> str:
        """
        Ask a clarifying question to better understand the client's needs.
        """
        logger.info(f"Asking clarification: {question}")
        return question

    # ------------------
    # Function: Schedule Follow-up
    # ------------------
    @function_tool()
    async def schedule_followup(self, reason: str = "discuss solutions in detail") -> str:
        """
        Offer to connect the client with a Tekisho expert.
        """
        name = self.conversation_context.get("client_name", "there")
        company = self.conversation_context.get("company", "your company")
        
        return (f"I'd love to connect you with one of our solution architects who can {reason} "
               f"specifically for {company}. They'll provide a customized proposal and answer "
               f"any technical questions you might have. Would that be helpful, {name}?")

    # ------------------
    # Function: Summarize Conversation
    # ------------------
    @function_tool()
    async def summarize_conversation(self) -> str:
        """
        Provide a summary of what was discussed and next steps.
        """
        name = self.conversation_context.get("client_name", "you")
        company = self.conversation_context.get("company", "your organization")
        challenges = self.conversation_context.get("challenges_discussed", [])
        
        if challenges:
            summary = (f"It's been great talking with you, {name}! We've discussed how Tekisho can help {company} "
                      f"with {', '.join(challenges[:2])}. ")
        else:
            summary = f"Thank you for sharing about {company}'s goals. "
        
        summary += ("I can connect you with our team to dive deeper into solutions, "
                   "provide specific ROI calculations, and discuss implementation timelines. "
                   "Would you like me to arrange that?")
        
        return summary


# =====================================
# Entrypoint
# =====================================
async def entrypoint(ctx: agents.JobContext):
    """Main entry for LiveKit agent session."""
    logger.info("Starting Tekisho RAG-Powered Agent with DB Integration...")

    # Create agent with instructions from prompts.py
    agent = Assistant(instructions=AGENT_INSTRUCTION)

    # Set up voice, avatar, and session
    session = AgentSession(
        llm="openai/gpt-4o-mini",  # Using GPT-4 for better function calling
        stt="assemblyai/universal-streaming",
        tts="cartesia/sonic-2:9626c31c-bec5-4cca-baa8-f8ba9e84c8bc",
        vad=silero.VAD.load(),
    )

    avatar = tavus.AvatarSession(
        replica_id=REPLICA_ID,
        persona_id=PERSONA_ID,
        api_key=TAVUS_API_KEY,
    )

    try:
        await avatar.start(session, room=ctx.room)

        await session.start(
            room=ctx.room,
            agent=agent,
            room_input_options=RoomInputOptions(
                noise_cancellation=noise_cancellation.BVC(),
            ),
        )

        # Generate initial greeting
        await session.generate_reply(instructions=SESSION_INSTRUCTION)
        
    except Exception as e:
        logger.error(f"Error in agent session: {e}")
        # Ensure avatar session is properly cleaned up on error
        try:
            if hasattr(avatar, 'conversation_id') and avatar.conversation_id:
                logger.info("Cleaning up Tavus avatar session due to error...")
                await avatar.stop()
        except Exception as cleanup_error:
            logger.warning(f"Error during avatar cleanup: {cleanup_error}")
        raise
    finally:
        # Always try to cleanup avatar session when done
        try:
            if hasattr(avatar, 'conversation_id') and avatar.conversation_id:
                logger.info("Cleaning up Tavus avatar session...")
                await avatar.stop()
        except Exception as cleanup_error:
            logger.warning(f"Error during final avatar cleanup: {cleanup_error}")


# =====================================
# CLI Run
# =====================================
if __name__ == "_main_":
    logger.info("Launching Tekisho RAG-Powered Assistant with DB Integration...")
    agents.cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            ws_url=LIVEKIT_URL,
            api_key=LIVEKIT_API_KEY,
            api_secret=LIVEKIT_API_SECRET,
        ),
    )