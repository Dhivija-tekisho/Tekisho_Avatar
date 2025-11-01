import os
import json
import logging
import asyncio
from livekit import api
from flask import Flask, request, jsonify
from dotenv import load_dotenv
from flask_cors import CORS
from livekit.api import LiveKitAPI, ListRoomsRequest
from supabase_client import get_supabase_client, format_chat_message
import uuid

load_dotenv()

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("TekishoServer")

async def generate_room_name():
    name = "room-" + str(uuid.uuid4())[:8]
    rooms = await get_rooms()
    while name in rooms:
        name = "room-" + str(uuid.uuid4())[:8]
    return name

async def get_rooms():
    api = LiveKitAPI()
    rooms = await api.room.list_rooms(ListRoomsRequest())
    await api.aclose()
    return [room.name for room in rooms.rooms]

@app.route("/getToken")
async def get_token():
    name = request.args.get("name", "my name")
    room = request.args.get("room", None)
    
    if not room:
        room = await generate_room_name()
        
    token = api.AccessToken(os.getenv("LIVEKIT_API_KEY"), os.getenv("LIVEKIT_API_SECRET")) \
        .with_identity(name)\
        .with_name(name)\
        .with_grants(api.VideoGrants(
            room_join=True,
            room=room
        ))
    
    return token.to_jwt()

@app.route("/save_chat", methods=["POST"])
async def save_chat():
    """
    Save chat history to Supabase when conversation ends.
    
    Expected JSON payload:
    {
        "name": "Client Name",
        "company_name": "Company Name", 
        "chat_history": [
            {
                "timestamp": "2024-01-01T12:00:00",
                "speaker": "User/Agent/System",
                "message": "Chat message content",
                "type": "text"
            }
        ]
    }
    """
    try:
        # Get JSON data from request
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
        
        # Validate required fields
        name = data.get("name", "Unknown")
        company_name = data.get("company_name", "Unknown Company")
        chat_history = data.get("chat_history", [])
        
        if not chat_history:
            return jsonify({"error": "No chat history provided"}), 400
        
        logger.info(f"Received request to save chat for {name} from {company_name} with {len(chat_history)} messages")
        
        # Get Supabase client and save chat
        supabase_client = get_supabase_client()
        result = await supabase_client.save_chat_history(name, company_name, chat_history)
        
        if "error" in result:
            logger.error(f"Failed to save chat: {result['error']}")
            return jsonify({"error": result["error"]}), 500
        
        logger.info(f"Successfully saved chat history for {name}")
        return jsonify({
            "success": True,
            "message": f"Chat history saved successfully for {name}",
            "record_id": result.get("id"),
            "message_count": len(chat_history)
        })
        
    except Exception as e:
        logger.error(f"Error in save_chat endpoint: {str(e)}")
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500

@app.route("/get_chats", methods=["GET"])
async def get_chats():
    """
    Retrieve chat history from Supabase.
    
    Query parameters:
    - name: Optional filter by client name
    - company_name: Optional filter by company name
    - limit: Maximum number of records (default 50)
    """
    try:
        # Get query parameters
        name = request.args.get("name")
        company_name = request.args.get("company_name")
        limit = int(request.args.get("limit", 50))
        
        logger.info(f"Retrieving chats with filters: name={name}, company={company_name}, limit={limit}")
        
        # Get Supabase client and retrieve chats
        supabase_client = get_supabase_client()
        chats = await supabase_client.get_chat_history(name, company_name, limit)
        
        logger.info(f"Retrieved {len(chats)} chat records")
        return jsonify({
            "success": True,
            "chats": chats,
            "count": len(chats)
        })
        
    except Exception as e:
        logger.error(f"Error in get_chats endpoint: {str(e)}")
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500

@app.route("/extract_client_info", methods=["POST"])
async def extract_client_info():
    """
    Use LLM to extract client name and company from chat history.
    
    Expected JSON payload:
    {
        "chat_history": [
            {
                "speaker": "User/Agent",
                "message": "conversation text"
            }
        ]
    }
    """
    try:
        # Get JSON data from request
        data = request.get_json()
        if not data or not data.get("chat_history"):
            return jsonify({"error": "No chat history provided"}), 400
        
        chat_history = data.get("chat_history", [])
        
        # Prepare conversation text for LLM
        conversation_text = ""
        for msg in chat_history:
            speaker = msg.get("speaker", "Unknown")
            message = msg.get("message", "")
            conversation_text += f"{speaker}: {message}\n"
        
        # Use OpenAI to extract client information
        from openai import OpenAI
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        prompt = f"""
        Analyze the following conversation and extract the client's name and company name.
        Return ONLY a JSON object with 'name' and 'company' fields.
        If information is not found, use 'Unknown' for that field.
        
        Conversation:
        {conversation_text}
        
        Example response format:
        {{"name": "John Doe", "company": "Acme Corp"}}
        """
        
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=100,
            temperature=0.1
        )
        
        # Parse the LLM response
        result_text = response.choices[0].message.content.strip()
        
        try:
            # Try to parse as JSON
            import json
            extracted_info = json.loads(result_text)
            name = extracted_info.get("name", "Unknown")
            company = extracted_info.get("company", "Unknown")
        except:
            # Fallback if JSON parsing fails
            name = "Unknown"
            company = "Unknown"
        
        logger.info(f"Extracted client info: {name} from {company}")
        
        return jsonify({
            "success": True,
            "name": name,
            "company": company
        })
        
    except Exception as e:
        logger.error(f"Error in extract_client_info endpoint: {str(e)}")
        return jsonify({"error": f"Failed to extract client info: {str(e)}"}), 500

@app.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint."""
    return jsonify({"status": "healthy", "service": "Tekisho Chat API"})

@app.route("/save-conversation", methods=["POST"])
def save_conversation():
    """
    Endpoint to save conversation transcript to database.
    Extracts user name and company using LLM.
    """
    try:
        data = request.get_json()
        
        if not data or 'chat_history' not in data:
            return jsonify({"error": "No chat history provided"}), 400
        
        chat_history = data['chat_history']
        
        if not chat_history:
            return jsonify({"error": "Chat history is empty"}), 400
        
        logger.info(f"📝 Processing conversation save - {len(chat_history)} messages")
        
        # Import from separate extractor module (avoids LiveKit plugin issues)
        from llm_extractor import extract_user_info_from_chat
        
        # Extract user info using LLM (synchronous)
        user_info = extract_user_info_from_chat(chat_history)
        
        name = user_info.get('name', 'Unknown')
        company = user_info.get('company', 'Unknown')
        
        logger.info(f"📊 Extracted info - Name: {name}, Company: {company}")
        
        # Save to Supabase synchronously
        supabase_client = get_supabase_client()
        
        # Call the async function synchronously using asyncio
        import asyncio
        try:
            # Try to get existing event loop
            try:
                loop = asyncio.get_event_loop()
                if loop.is_closed():
                    raise RuntimeError("Loop is closed")
            except RuntimeError:
                # Create new event loop if none exists
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            
            # Run the async function
            if loop.is_running():
                # If loop is already running, create a task
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as pool:
                    result = pool.submit(
                        asyncio.run,
                        supabase_client.save_chat_history(
                            name=name,
                            company_name=company,
                            chat_history=chat_history
                        )
                    ).result()
            else:
                # If loop is not running, run it
                result = loop.run_until_complete(
                    supabase_client.save_chat_history(
                        name=name,
                        company_name=company,
                        chat_history=chat_history
                    )
                )
        except Exception as loop_error:
            logger.error(f"Event loop error: {loop_error}")
            # Fallback: create fresh loop
            new_loop = asyncio.new_event_loop()
            result = new_loop.run_until_complete(
                supabase_client.save_chat_history(
                    name=name,
                    company_name=company,
                    chat_history=chat_history
                )
            )
            new_loop.close()
        
        logger.info(f"💾 Saved conversation - Name: {name}, Company: {company}, Messages: {len(chat_history)}")
        
        return jsonify({
            "success": True,
            "name": name,
            "company": company,
            "message_count": len(chat_history),
            "record_id": result.get('id') if result else None
        })
        
    except Exception as e:
        logger.error(f"Error saving conversation: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)