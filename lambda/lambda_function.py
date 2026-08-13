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

# Amazon Bedrock configuration
bedrock_model_id = os.environ.get("BEDROCK_MODEL_ID", "us.anthropic.claude-3-5-haiku-20241022-v1:0")
aws_access_key_id = os.environ.get("AWS_ACCESS_KEY_ID", os.environ.get("BEDROCK_ACCESS_KEY", ""))
aws_secret_access_key = os.environ.get("AWS_SECRET_ACCESS_KEY", os.environ.get("BEDROCK_SECRET_KEY", ""))
aws_region = os.environ.get("AWS_REGION", os.environ.get("AWS_DEFAULT_REGION", "us-east-1"))
bedrock_api_key = os.environ.get("BEDROCK_API_KEY", os.environ.get("AWS_BEARER_TOKEN", ""))

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

import boto3

# Amazon Bedrock configuration
bedrock_model_id = os.environ.get("BEDROCK_MODEL_ID", "us.anthropic.claude-3-5-haiku-20241022-v1:0")
aws_access_key_id = os.environ.get("AWS_ACCESS_KEY_ID", os.environ.get("BEDROCK_ACCESS_KEY", ""))
aws_secret_access_key = os.environ.get("AWS_SECRET_ACCESS_KEY", os.environ.get("BEDROCK_SECRET_KEY", ""))
aws_region = os.environ.get("AWS_REGION", os.environ.get("AWS_DEFAULT_REGION", "us-east-1"))
bedrock_api_key = os.environ.get("BEDROCK_API_KEY", os.environ.get("AWS_BEARER_TOKEN", ""))

def generate_ai_response(chat_history, new_question, is_followup=False):
    """Generates an AI response using Amazon Bedrock (Claude Haiku 3.5 / Amazon Nova)."""
    system_message = (
        "You are an AI voice assistant responding out loud through an Alexa device. "
        "Follow these strict rules for your response format:\n"
        "1. Write exclusively for SPOKEN voice output (Text-To-Speech). Use natural, conversational, fluid prose that sounds great when spoken out loud.\n"
        "2. Do NOT use markdown syntax, headers (#), bullet points (-), numbered lists, bold (**), italics (*), or visual list formatting.\n"
        "3. Keep answers clear, engaging, and concise (about 2 to 4 sentences, or a well-paced single short paragraph). Provide rich detail without being overly verbose or overwhelming for a listener."
    )
    if is_followup:
        system_message += " This is a follow-up question. Answer directly and concisely, building naturally on the context."
    
    history_limit = 10 if not is_followup else 5
    
    # Models to try in order of preference
    models_to_try = [
        bedrock_model_id,
        "us.anthropic.claude-3-haiku-20240307-v1:0",
        "us.amazon.nova-micro-v1:0",
        "us.amazon.nova-lite-v1:0"
    ]
    
    for model_id in models_to_try:
        try:
            logger.info(f"Attempting Amazon Bedrock model '{model_id}' in region '{aws_region}'...")
            
            # Format payload based on model family
            if "nova" in model_id:
                nova_messages = []
                for question, answer in chat_history[-history_limit:]:
                    nova_messages.append({"role": "user", "content": [{"text": question}]})
                    nova_messages.append({"role": "assistant", "content": [{"text": answer}]})
                nova_messages.append({"role": "user", "content": [{"text": new_question}]})
                
                payload = {
                    "inferenceConfig": {"max_new_tokens": 500, "temperature": 0.7},
                    "system": [{"text": system_message}],
                    "messages": nova_messages
                }
            else:
                claude_messages = []
                for question, answer in chat_history[-history_limit:]:
                    claude_messages.append({"role": "user", "content": question})
                    claude_messages.append({"role": "assistant", "content": answer})
                claude_messages.append({"role": "user", "content": new_question})
                
                payload = {
                    "anthropic_version": "bedrock-2023-05-31",
                    "max_tokens": 500,
                    "temperature": 0.7,
                    "system": system_message,
                    "messages": claude_messages
                }

            # Method 1: Bearer Token (HTTP) if provided
            if bedrock_api_key:
                url = f"https://bedrock-runtime.{aws_region}.amazonaws.com/model/{model_id}/invoke"
                headers = {
                    "Authorization": f"Bearer {bedrock_api_key}",
                    "Content-Type": "application/json",
                    "Accept": "application/json"
                }
                res = requests.post(url, headers=headers, json=payload, timeout=12)
                if res.ok:
                    res_data = res.json()
                    if "nova" in model_id:
                        raw_text = res_data.get("output", {}).get("message", {}).get("content", [{}])[0].get("text", "")
                    else:
                        raw_text = res_data.get("content", [{}])[0].get("text", "")
                    if raw_text:
                        logger.info(f"Successfully generated response with Bedrock model '{model_id}'")
                        return clean_text_for_speech(raw_text), ["Tell me more", "Explain in detail"]
                else:
                    logger.warning(f"Bedrock model '{model_id}' HTTP returned status {res.status_code}: {res.text[:150]}")

            # Method 2: boto3 SDK
            client_kwargs = {"region_name": aws_region}
            if aws_access_key_id and aws_secret_access_key:
                client_kwargs["aws_access_key_id"] = aws_access_key_id
                client_kwargs["aws_secret_access_key"] = aws_secret_access_key
            
            bedrock_runtime = boto3.client("bedrock-runtime", **client_kwargs)
            response = bedrock_runtime.invoke_model(
                modelId=model_id,
                contentType="application/json",
                accept="application/json",
                body=json.dumps(payload)
            )
            response_body = json.loads(response["body"].read().decode("utf-8"))
            if "nova" in model_id:
                raw_text = response_body.get("output", {}).get("message", {}).get("content", [{}])[0].get("text", "")
            else:
                raw_text = response_body.get("content", [{}])[0].get("text", "")
            if raw_text:
                logger.info(f"Successfully generated response with boto3 Bedrock model '{model_id}'")
                return clean_text_for_speech(raw_text), ["Tell me more", "Explain in detail"]
        except Exception as e:
            logger.warning(f"Amazon Bedrock call for model '{model_id}' failed: {e}")

    return "I'm sorry, I'm having trouble connecting to Amazon Bedrock right now. Please verify your Amazon Bedrock credentials.", ["Try again"]

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
