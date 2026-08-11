import os
from ask_sdk_core.dispatch_components import AbstractExceptionHandler
from ask_sdk_core.dispatch_components import AbstractRequestHandler
from ask_sdk_core.skill_builder import SkillBuilder
from ask_sdk_core.handler_input import HandlerInput
from ask_sdk_model import Response
import ask_sdk_core.utils as ask_utils
import requests
import logging
import json
import re
import time

# Load .env file if present locally or bundled in Lambda
env_file_path = os.path.join(os.path.dirname(__file__), ".env")
if os.path.exists(env_file_path):
    with open(env_file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ[k.strip()] = v.strip()

poe_api_key = os.environ.get("POE_API_KEY", "")
openrouter_api_key = os.environ.get("OPENROUTER_API_KEY", "")
primary_model = os.environ.get("PRIMARY_MODEL", "claude-haiku-4.5")
fallback_model = os.environ.get("FALLBACK_MODEL", "anthropic/claude-3-haiku")

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class LaunchRequestHandler(AbstractRequestHandler):
    """Handler for Skill Launch."""
    def can_handle(self, handler_input):
        # type: (HandlerInput) -> bool
        return ask_utils.is_request_type("LaunchRequest")(handler_input)

    def handle(self, handler_input):
        # type: (HandlerInput) -> Response
        speak_output = "Claude AI mode activated"

        session_attr = handler_input.attributes_manager.session_attributes
        session_attr["chat_history"] = []

        return (
            handler_input.response_builder
                .speak(speak_output)
                .ask(speak_output)
                .response
        )

class HelpIntentHandler(AbstractRequestHandler):
    """Handler for Help Intent."""
    def can_handle(self, handler_input):
        # type: (HandlerInput) -> bool
        return ask_utils.is_intent_name("AMAZON.HelpIntent")(handler_input)

    def handle(self, handler_input):
        # type: (HandlerInput) -> Response
        speak_output = "Claude AI mode activated. You can ask me any question you like. What would you like to know?"

        return (
            handler_input.response_builder
                .speak(speak_output)
                .ask(speak_output)
                .response
        )

class ClaudeQueryIntentHandler(AbstractRequestHandler):
    """Handler for Claude Query Intent, Gemini Query Intent, Fallback Intent, and Sample Intents."""
    def can_handle(self, handler_input):
        # type: (HandlerInput) -> bool
        return (ask_utils.is_intent_name("ClaudeQueryIntent")(handler_input) or 
                ask_utils.is_intent_name("GeminiQueryIntent")(handler_input) or 
                ask_utils.is_intent_name("GptQueryIntent")(handler_input) or
                ask_utils.is_intent_name("HelloWorldIntent")(handler_input) or
                ask_utils.is_intent_name("AMAZON.FallbackIntent")(handler_input))

    def handle(self, handler_input):
        # type: (HandlerInput) -> Response
        query = None
        req = handler_input.request_envelope.request
        if hasattr(req, "intent") and req.intent and hasattr(req.intent, "slots") and req.intent.slots:
            if "query" in req.intent.slots and req.intent.slots["query"].value:
                query = req.intent.slots["query"].value
                
        if not query:
            query = "Hello, what can you do?"

        session_attr = handler_input.attributes_manager.session_attributes
        if "chat_history" not in session_attr:
            session_attr["chat_history"] = []
            session_attr["last_context"] = None
        
        processed_query, is_followup = process_followup_question(query, session_attr.get("last_context"))
        
        response_data = generate_claude_response(session_attr["chat_history"], processed_query, is_followup)
        
        if isinstance(response_data, tuple) and len(response_data) == 2:
            response_text, followup_questions = response_data
        else:
            response_text = str(response_data)
            followup_questions = []
        
        session_attr["followup_questions"] = followup_questions
        session_attr["chat_history"].append((query, response_text))
        session_attr["last_context"] = extract_context(query, response_text)
        
        response = response_text
        if followup_questions and len(followup_questions) > 0:
            response += " <break time=\"0.5s\"/> "
            response += "You could ask: "
            if len(followup_questions) > 1:
                response += ", ".join([f"'{q}'" for q in followup_questions[:-1]])
                response += f", or '{followup_questions[-1]}'"
            else:
                response += f"'{followup_questions[0]}'"
            response += ". <break time=\"0.5s\"/> What would you like to know?"
        
        reprompt_text = "You can ask me another question or say stop to end the conversation."
        if 'followup_questions' in session_attr and session_attr['followup_questions']:
            reprompt_text = "You can ask me another question, say 'next' to hear more suggestions, or say stop to end the conversation."
        
        return (
            handler_input.response_builder
                .speak(response)
                .ask(reprompt_text)
                .response
        )

# Aliases for compatibility
GeminiQueryIntentHandler = ClaudeQueryIntentHandler
GptQueryIntentHandler = ClaudeQueryIntentHandler

class CatchAllExceptionHandler(AbstractExceptionHandler):
    """Generic error handling to capture any syntax or routing errors."""
    def can_handle(self, handler_input, exception):
        # type: (HandlerInput, Exception) -> bool
        return True

    def handle(self, handler_input, exception):
        # type: (HandlerInput, Exception) -> Response
        logger.error(f"Error caught in CatchAllExceptionHandler: {str(exception)}", exc_info=True)

        speak_output = "Sorry, I had trouble doing what you asked. Please try again."

        return (
            handler_input.response_builder
                .speak(speak_output)
                .ask(speak_output)
                .response
        )

class CancelOrStopIntentHandler(AbstractRequestHandler):
    """Single handler for Cancel and Stop Intent."""
    def can_handle(self, handler_input):
        # type: (HandlerInput) -> bool
        return (ask_utils.is_intent_name("AMAZON.CancelIntent")(handler_input) or
                ask_utils.is_intent_name("AMAZON.StopIntent")(handler_input))

    def handle(self, handler_input):
        # type: (HandlerInput) -> Response
        speak_output = "Leaving Claude mode"

        return (
            handler_input.response_builder
                .speak(speak_output)
                .response
        )

def process_followup_question(question, last_context):
    """Processes a question to determine if it's a follow-up and enhances it with context if needed"""
    followup_patterns = [
        r'^(what|how|why|when|where|who|which)\s+(about|is|are|was|were|do|does|did|can|could|would|should|will)\s',
        r'^(and|but|so|then|also)\s',
        r'^(can|could|would|should|will)\s+(you|it|they|we)\s',
        r'^(is|are|was|were|do|does|did)\s+(it|that|this|they|those|these)\s',
        r'^(tell me more|elaborate|explain further)\s*',
        r'^(why|how)\?*$'
    ]
    
    is_followup = False
    for pattern in followup_patterns:
        if re.search(pattern, question.lower()):
            is_followup = True
            break
    
    return question, is_followup

def clean_text_for_speech(text):
    """Sanitizes text so Alexa reads it naturally out loud without markdown formatting artifacts."""
    if not text:
        return ""
    # Strip stage direction prefixes (e.g. "in a friendly, conversational tone", "[spoken out loud]")
    text = re.sub(r'^(?:in a\s+[a-z\s,]+tone|\[[^\]]+\]|\([^\)]+\))\s*', '', text, flags=re.IGNORECASE)
    # Strip markdown headers (### Header -> Header)
    text = re.sub(r'#+\s*', '', text)
    # Strip bold/italic markdown (**text**, *text*, __text__, _text_)
    text = re.sub(r'[\*_]{1,3}([^\*_]+)[\*_]{1,3}', r'\1', text)
    # Strip bullet points and list markers at start of line
    text = re.sub(r'^\s*[\-\*\+]\s+', '', text, flags=re.MULTILINE)
    text = re.sub(r'^\s*\d+\.\s+', '', text, flags=re.MULTILINE)
    # Strip code block backticks
    text = re.sub(r'`{1,3}[^`]*`{1,3}', '', text)
    # Convert multiple spaces or newlines to single space
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def extract_context(question, response):
    """Extracts the main context from a Q&A pair for future reference"""
    return {"question": question, "response": response}

def generate_followup_questions(conversation_context, query, response_text, count=2):
    """Returns concise follow-up question suggestions without extra API calls to save quota."""
    return ["Tell me more", "Explain in detail"]

def generate_ai_response(chat_history, new_question, is_followup=False):
    """Generates an AI response using Poe API (GPT-5.6-Luna) or OpenRouter fallback."""
    system_message = (
        "You are an AI voice assistant responding out loud through an Alexa device. "
        "Follow these strict rules for your response format:\n"
        "1. Write exclusively for SPOKEN voice output (Text-To-Speech). Use natural, conversational, fluid prose that sounds great when spoken out loud.\n"
        "2. Do NOT use markdown syntax, headers (#), bullet points (-), numbered lists, bold (**), italics (*), or visual list formatting.\n"
        "3. Keep answers clear, engaging, and concise (about 2 to 4 sentences, or a well-paced single short paragraph). Provide rich detail without being overly verbose or overwhelming for a listener."
    )
    if is_followup:
        system_message += " This is a follow-up question. Answer directly and concisely, building naturally on the context."
    
    messages = [{"role": "system", "content": system_message}]
    
    # Include relevant conversation history
    history_limit = 10 if not is_followup else 5
    for question, answer in chat_history[-history_limit:]:
        messages.append({"role": "user", "content": question})
        messages.append({"role": "assistant", "content": answer})
    
    # Add the new question
    messages.append({"role": "user", "content": new_question})
    
    # Attempt 1: Poe API
    if poe_api_key:
        poe_url = "https://api.poe.com/v1/chat/completions"
        poe_headers = {
            "Authorization": f"Bearer {poe_api_key}",
            "Content-Type": "application/json"
        }
        poe_payload = {
            "model": primary_model,
            "messages": messages,
            "temperature": 0.7
        }
        try:
            logger.info(f"Attempting Poe API with model {primary_model}...")
            res = requests.post(poe_url, headers=poe_headers, json=poe_payload, timeout=15)
            if res.ok:
                res_data = res.json()
                choices = res_data.get('choices', [])
                if choices:
                    raw_text = choices[0].get('message', {}).get('content', '')
                    return clean_text_for_speech(raw_text), ["Tell me more", "Explain in detail"]
            else:
                logger.warning(f"Poe API returned status {res.status_code}: {res.text[:200]}")
        except Exception as e:
            logger.error(f"Error calling Poe API: {e}")

    # Attempt 2: OpenRouter Fallback
    if openrouter_api_key:
        or_url = "https://openrouter.ai/api/v1/chat/completions"
        or_headers = {
            "Authorization": f"Bearer {openrouter_api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/MatLudke/ClaudeInAlexa",
            "X-Title": "Alexa AI Skill"
        }
        or_payload = {
            "model": fallback_model,
            "messages": messages,
            "max_tokens": 2048,
            "temperature": 0.7
        }
        try:
            logger.info(f"Attempting OpenRouter API fallback with model {fallback_model}...")
            res = requests.post(or_url, headers=or_headers, json=or_payload, timeout=15)
            if res.ok:
                res_data = res.json()
                choices = res_data.get('choices', [])
                if choices:
                    raw_text = choices[0].get('message', {}).get('content', '')
                    return clean_text_for_speech(raw_text), ["Tell me more", "Explain in detail"]
            else:
                logger.warning(f"OpenRouter API returned status {res.status_code}: {res.text[:200]}")
        except Exception as e:
            logger.error(f"Error calling OpenRouter API: {e}")

    return "I'm sorry, I'm having trouble connecting to the AI model right now. Please check your Poe subscription or API key.", ["Try again"]

generate_claude_response = generate_ai_response
generate_gemini_response = generate_ai_response
generate_gpt_response = generate_ai_response

class ClearContextIntentHandler(AbstractRequestHandler):
    """Handler for clearing conversation context."""
    def can_handle(self, handler_input):
        # type: (HandlerInput) -> bool
        return ask_utils.is_intent_name("ClearContextIntent")(handler_input)

    def handle(self, handler_input):
        # type: (HandlerInput) -> Response
        session_attr = handler_input.attributes_manager.session_attributes
        session_attr["chat_history"] = []
        session_attr["last_context"] = None
        
        speak_output = "I've cleared our conversation history. What would you like to talk about?"
        
        return (
            handler_input.response_builder
                .speak(speak_output)
                .ask(speak_output)
                .response
        )

sb = SkillBuilder()

sb.add_request_handler(LaunchRequestHandler())
sb.add_request_handler(HelpIntentHandler())
sb.add_request_handler(ClaudeQueryIntentHandler())
sb.add_request_handler(ClearContextIntentHandler())
sb.add_request_handler(CancelOrStopIntentHandler())
sb.add_exception_handler(CatchAllExceptionHandler())

lambda_handler = sb.lambda_handler()
