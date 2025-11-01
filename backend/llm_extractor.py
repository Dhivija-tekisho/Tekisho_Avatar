"""
LLM-based extraction utilities for chat analysis.
Separate from agent.py to avoid LiveKit plugin registration issues.
"""
import os
import json
import logging
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger("LLMExtractor")


def extract_user_info_from_chat(chat_history: list) -> dict:
    """
    Use LLM to extract user's name and company from chat history.
    Filters out greetings like 'Hi Aria' to avoid capturing 'Aria' as the user's name.
    
    Args:
        chat_history: List of transcript messages
        
    Returns:
        Dict with 'name' and 'company' keys
    """
    try:
        from openai import OpenAI
        
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        # Format chat history for LLM - only last 20 messages for speed
        recent_messages = chat_history[-20:] if len(chat_history) > 20 else chat_history
        
        conversation_text = "\n".join([
            f"{msg.get('speaker', 'Unknown')}: {msg.get('message', '')}" 
            for msg in recent_messages
            if msg.get('type') != 'system'  # Exclude system messages
        ])
        
        if not conversation_text.strip():
            logger.warning("No conversation text to analyze")
            return {"name": "Unknown", "company": "Unknown"}
        
        # Create shorter, more focused prompt for faster extraction
        extraction_prompt = f"""Extract the user's name and company from this conversation.

RULES:
1. NEVER extract "Aria" as the user's name (Aria is the AI assistant)
2. Ignore greetings like "Hi Aria", "Hello Aria"
3. Look for: "I'm [name]", "My name is [name]", "from [company]", "work at [company]"
4. Return "Unknown" if information is not clearly stated

Conversation:
{conversation_text[:1000]}

Respond ONLY with JSON in this exact format:
{{"name": "extracted name or Unknown", "company": "extracted company or Unknown"}}
"""
        
        logger.info("🔍 Extracting user info from conversation...")
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system", 
                    "content": "You are a data extraction assistant. Extract user information accurately. NEVER extract 'Aria' as the user's name. Return JSON only."
                },
                {"role": "user", "content": extraction_prompt}
            ],
            temperature=0,
            max_tokens=100,  # Limit tokens for faster response
            response_format={"type": "json_object"}
        )
        
        result = json.loads(response.choices[0].message.content)
        
        # Validate Aria is not captured
        name = result.get('name', 'Unknown')
        company = result.get('company', 'Unknown')
        
        if name.lower() in ['aria', 'hi aria', 'hello aria', 'hey aria', 'hi', 'hello']:
            logger.warning(f"Filtered out invalid name: {name}")
            name = 'Unknown'
        
        logger.info(f"✅ Extracted - Name: {name}, Company: {company}")
        return {"name": name, "company": company}
        
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse LLM response as JSON: {e}")
        return {"name": "Unknown", "company": "Unknown"}
    except Exception as e:
        logger.error(f"Failed to extract user info: {e}")
        return {"name": "Unknown", "company": "Unknown"}
