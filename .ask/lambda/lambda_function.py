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

# Set your OpenRouter API key
api_key = os.environ.get("OPENROUTER_API_KEY", os.environ.get("CLAUDE_API_KEY", "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"))

primary_model = "anthropic/claude-3-haiku"
fallback_model = "anthropic/claude-3.5-haiku"

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class LaunchRequestHandler(AbstractRequestHandler):
    """Handler for Skill Launch."""
    def can_handle(self, handler_input):
        # type: (HandlerInput) -> bool
        return ask_utils.is_request_type("LaunchRequest")(handler_input)

    def handle(self, handler_input):
        # type: (HandlerInput) -> Response
        speak_output = "Claude mode activated"

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
        speak_output = "Claude mode activated. You can ask me any question you like. What would you like to know?"

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

def extract_context(question, response):
    """Extracts the main context from a Q&A pair for future reference"""
    return {"question": question, "response": response}

def generate_followup_questions(conversation_context, query, response_text, count=2):
    """Returns concise follow-up question suggestions without extra API calls to save quota."""
    return ["Tell me more", "Explain in detail"]

def generate_claude_response(chat_history, new_question, is_followup=False):
    """Generates a Claude response via OpenRouter API with retry and fallback handling."""
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/MatLudke/ClaudeInAlexa",
        "X-Title": "Claude Alexa Skill"
    }
    
    system_message = "You are a helpful assistant powered by Claude. Provide clear, comprehensive, and up-to-date answers. Feel free to explain in detail."
    if is_followup:
        system_message += " This is a follow-up question to the previous conversation. Maintain context without repeating information already provided."
    
    messages = [{"role": "system", "content": system_message}]
    
    # Include relevant conversation history
    history_limit = 10 if not is_followup else 5
    for question, answer in chat_history[-history_limit:]:
        messages.append({"role": "user", "content": question})
        messages.append({"role": "assistant", "content": answer})
    
    # Add the new question
    messages.append({"role": "user", "content": new_question})
    
    payload = {
        "model": primary_model,
        "messages": messages,
        "max_tokens": 2048,
        "temperature": 0.7
    }
    
    try:
        response = None
        for attempt in range(2):
            response = requests.post(url, headers=headers, data=json.dumps(payload), timeout=15)
            if response.ok:
                break
            elif response.status_code in [429, 502, 503, 504]:
                logger.warning(f"Got status {response.status_code} on attempt {attempt+1}. Sleeping 1.5s...")
                time.sleep(1.5)
            else:
                break
                
        # If primary failed, try fallback model
        if not response or not response.ok:
            logger.warning(f"Primary model {primary_model} failed (Status: {response.status_code if response else 'None'}). Falling back to {fallback_model}...")
            payload["model"] = fallback_model
            response = requests.post(url, headers=headers, data=json.dumps(payload), timeout=15)
            
        response_data = response.json()
        if response.ok:
            choices = response_data.get('choices', [])
            if choices:
                response_text = choices[0].get('message', {}).get('content', '')
            else:
                response_text = "No response choices returned from Claude."
            
            followup_questions = generate_followup_questions(
                chat_history + [(new_question, response_text)], 
                new_question, 
                response_text
            )
            return response_text, followup_questions
        else:
            error_msg = response_data.get('error', {}).get('message', response.text)
            logger.error(f"OpenRouter Error ({response.status_code}): {error_msg}")
            return f"Error {response.status_code}: {error_msg}", []
    except Exception as e:
        logger.error(f"Error generating response: {str(e)}")
        return f"Error generating response: {str(e)}", []

generate_gemini_response = generate_claude_response
generate_gpt_response = generate_claude_response

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
