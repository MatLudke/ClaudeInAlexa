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

import urllib.parse
import xml.etree.ElementTree as ET
import html
from concurrent.futures import ThreadPoolExecutor

def extract_main_article_text(html_content, max_chars=1200):
    """Strips HTML tags, script, and style tags to extract clean main article body text."""
    clean = re.sub(r'<(script|style|head|footer|nav|header)[^>]*>.*?</\1>', '', html_content, flags=re.DOTALL | re.IGNORECASE)
    clean = re.sub(r'<[^>]+>', ' ', clean)
    clean = html.unescape(clean)
    clean = re.sub(r'\s+', ' ', clean).strip()
    return clean[:max_chars]

def scrape_single_website(target_url, timeout=3):
    """Visits a website URL and extracts readable page text."""
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
    }
    try:
        res = requests.get(target_url, headers=headers, timeout=timeout, allow_redirects=True)
        if res.ok and len(res.text) > 400:
            text = extract_main_article_text(res.text)
            if len(text) > 100:
                return {"url": res.url, "content": text}
    except Exception as e:
        logger.debug(f"Error scraping {target_url}: {e}")
    return None

def fetch_live_search(query, max_websites=3):
    """
    Claude App style web browsing:
    1. Discovers top matching website URLs.
    2. Concurrently visits and scrapes full page contents of those websites.
    3. Feeds readable website body content into Claude's reasoning context.
    """
    target_urls = []
    
    # Step 1: Discover top URLs from Bing News RSS
    try:
        rss_url = f"https://www.bing.com/news/search?q={urllib.parse.quote(query)}&format=rss"
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
        res = requests.get(rss_url, headers=headers, timeout=3)
        if res.ok:
            root = ET.fromstring(res.text)
            for item in root.findall(".//item"):
                link = item.findtext("link", "")
                parsed = urllib.parse.urlparse(link)
                params = urllib.parse.parse_qs(parsed.query)
                actual_url = params.get("url", [None])[0]
                if actual_url and actual_url not in target_urls:
                    target_urls.append(actual_url)
                elif link and "bing.com" not in link and link not in target_urls:
                    target_urls.append(link)
                if len(target_urls) >= max_websites:
                    break
    except Exception as e:
        logger.warning(f"RSS link discovery error: {e}")

    # Step 2: Fallback discover Wikipedia URLs
    if len(target_urls) < max_websites:
        try:
            wiki_url = f"https://en.wikipedia.org/w/api.php?action=query&list=search&srsearch={urllib.parse.quote(query)}&format=json&utf8=1"
            res = requests.get(wiki_url, timeout=3)
            if res.ok:
                hits = res.json().get("query", {}).get("search", [])
                for hit in hits:
                    page_title = hit.get("title", "").replace(" ", "_")
                    wiki_link = f"https://en.wikipedia.org/wiki/{urllib.parse.quote(page_title)}"
                    if wiki_link not in target_urls:
                        target_urls.append(wiki_link)
                    if len(target_urls) >= max_websites:
                        break
        except Exception as e:
            logger.warning(f"Wiki link discovery error: {e}")

    logger.info(f"Visiting and reading websites: {target_urls}")

    # Step 3: Concurrently visit websites & extract body content
    scraped_pages = []
    with ThreadPoolExecutor(max_workers=max_websites) as executor:
        futures = [executor.submit(scrape_single_website, url) for url in target_urls]
        for f in futures:
            result = f.result()
            if result:
                scraped_pages.append(result)

    # Step 4: Build formatted multi-website context block
    formatted_context = ""
    for idx, page in enumerate(scraped_pages, 1):
        formatted_context += f"\n--- [Website #{idx} Visited: {page['url']}] ---\n{page['content']}\n"

    return formatted_context

import boto3

# Amazon Bedrock configuration
bedrock_model_id = os.environ.get("BEDROCK_MODEL_ID", "us.anthropic.claude-3-7-sonnet-20250219-v1:0")
aws_access_key_id = os.environ.get("AWS_ACCESS_KEY_ID", os.environ.get("BEDROCK_ACCESS_KEY", ""))
aws_secret_access_key = os.environ.get("AWS_SECRET_ACCESS_KEY", os.environ.get("BEDROCK_SECRET_KEY", ""))
aws_region = os.environ.get("AWS_REGION", os.environ.get("AWS_DEFAULT_REGION", "us-east-1"))
bedrock_api_key = os.environ.get("BEDROCK_API_KEY", os.environ.get("AWS_BEARER_TOKEN", ""))

def generate_ai_response(chat_history, new_question, is_followup=False):
    """Generates an AI response using Claude Sonnet 5 / 3.7 with Medium Reasoning and Web Search on Bedrock."""
    system_message = (
        "You are Claude, an exceptionally intelligent, articulate, and sharp AI voice companion speaking out loud through Alexa.\n"
        "Your objective is to provide authoritative, highly insightful, and captivating answers that feel remarkably smart without ever sounding robotic, generic, or formulaic.\n\n"
        "Follow these strict directives:\n"
        "1. NO CONVERSATIONAL FILLER OR PREAMBLE: Never start with fluff like 'Sure!', 'Great question!', 'Here is what I found', or 'As an AI'. Jump directly into the single most important fact or insight in the very first sentence.\n"
        "2. HIGH INFORMATION DENSITY: Avoid high-level generic summaries. Include precise names, concrete mechanics, key dates, or key figures where relevant, explaining the 'why' and 'how' behind concepts.\n"
        "3. OPTIMIZED FOR EAR COMPREHENSION: Write purely for spoken audio (Text-To-Speech). Use natural speech rhythm, varied sentence lengths, and elegant transitions. Never use markdown, bullet points, asterisks, numbered lists, special symbols, or visual formatting.\n"
        "4. SYNTHESIZE SEARCH & REASONING: Seamlessly integrate internal reasoning and provided live web search context into a fluid, confident, and masterfully crafted spoken response.\n"
        "5. CONCISE YET RICH DURATION: Keep responses to roughly 3 to 4 well-paced sentences (around 50 to 75 spoken words). Ensure every word delivers value."
    )
    if is_followup:
        system_message += " This is a follow-up question. Answer directly and concisely, building naturally on the context."
    
    # Perform live web search for context
    search_context = fetch_live_search(new_question)
    enhanced_question = new_question
    if search_context:
        logger.info("Enriched query with live web search results.")
        enhanced_question += f"\n\n[Live Web Search Context]:\n{search_context}"
    
    history_limit = 10 if not is_followup else 5
    
    # Models to try in order of preference (Claude Sonnet 5 / 3.7 -> Nova Pro / Lite)
    models_to_try = [
        bedrock_model_id,
        "us.anthropic.claude-3-7-sonnet-20250219-v1:0",
        "us.anthropic.claude-3-5-sonnet-20241022-v2:0",
        "us.amazon.nova-pro-v1:0",
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
                nova_messages.append({"role": "user", "content": [{"text": enhanced_question}]})
                
                payload = {
                    "inferenceConfig": {"max_new_tokens": 1024, "temperature": 0.7},
                    "system": [{"text": system_message}],
                    "messages": nova_messages
                }
            else:
                claude_messages = []
                for question, answer in chat_history[-history_limit:]:
                    claude_messages.append({"role": "user", "content": question})
                    claude_messages.append({"role": "assistant", "content": answer})
                claude_messages.append({"role": "user", "content": enhanced_question})
                
                # Payload with Extended Thinking / Reasoning at Medium (budget: 1024 tokens)
                payload = {
                    "anthropic_version": "bedrock-2023-05-31",
                    "max_tokens": 2048,
                    "temperature": 1.0,
                    "thinking": {
                        "type": "enabled",
                        "budget_tokens": 1024
                    },
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
                res = requests.post(url, headers=headers, json=payload, timeout=15)
                if res.ok:
                    res_data = res.json()
                    if "nova" in model_id:
                        raw_text = res_data.get("output", {}).get("message", {}).get("content", [{}])[0].get("text", "")
                    else:
                        # Extract SPOKEN text block only, excluding thinking reasoning block
                        raw_text = ""
                        for block in res_data.get("content", []):
                            if block.get("type") == "text":
                                raw_text += block.get("text", "")
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
                raw_text = ""
                for block in response_body.get("content", []):
                    if block.get("type") == "text":
                        raw_text += block.get("text", "")
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
