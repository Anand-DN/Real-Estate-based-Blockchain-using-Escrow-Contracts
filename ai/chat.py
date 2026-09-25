import json
import os
import re

import httpx
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), '.env'))

BASE_URL_OPENAI = 'https://api.groq.com/openai/v1'
OLLAMA_BASE = os.getenv('OLLAMA_URL', 'http://localhost:11434')
OLLAMA_API = f'{OLLAMA_BASE}/api/chat'

MODEL_GROQ = os.getenv('GROQ_MODEL', 'llama-3.3-70b-versatile')
MODEL_OLLAMA = os.getenv('OLLAMA_MODEL', 'llama3.1:8b')

DEFAULT_SYSTEM_PROMPT = (
    'You are Millow AI, the intelligent assistant inside the Millow Real Estate NFT DApp. '
    'You help users browse, value, buy and sell tokenized real estate on the Ethereum '
    'blockchain. Use a tool whenever the user asks for property insights, price predictions, '
    'fraud or risk analysis, similar properties, or market-wide statistics.\n\n'
    'Scope:\n'
    '- You ONLY answer questions related to this app: Millow listings, buying/selling, price '
    'predictions and valuations, risk/fraud analysis, escrow and blockchain mechanics, property '
    'metadata, and market statistics.\n'
    '- If the user asks about anything else (general programming, math, recipes, news, etc.), do '
    'NOT answer it. Politely decline in one short sentence and offer to help with the project '
    'listings or predictions instead.\n\n'
    'Response style:\n'
    '- Refer to properties by their name (e.g. "Luxury NYC Penthouse"). Never refer to a property '
    '"Property" followed by a number on its own, and never answer with just a token ID. If you must '
    'disambiguate, write the name first and add the token ID once in parentheses '
    '(e.g. "Luxury NYC Penthouse (property 5)").\n'
    '- Output plain text only. Do NOT use markdown, numbered points, bullet dashes, headings, or '
    'pipe tables.\n'
    '- Organize the answer as short paragraphs separated by blank lines, so it reads clean and '
    'breathable.\n'
    '- Medium length: roughly 3-5 short paragraphs covering the key details the tools returned '
    '(e.g. price, verdict, risk level with components, anomaly score, similar listings) when they '
    'fit the question.\n'
    '- Answer ONLY what the user asked. Do not drift into topics the user did not request.\n'
    '- Report prices in ETH. If a tool errors or a property is not found, say so plainly instead '
    'of inventing numbers.'
)


class ChatAgent:
    def __init__(self, tools, handlers, system_prompt=DEFAULT_SYSTEM_PROMPT, max_loops=6,
                 groq_api_key=None, request_timeout=90.0):
        self.tools = tools
        self.handlers = handlers
        self.system_prompt = system_prompt
        self.max_loops = max_loops
        self.groq_api_key = groq_api_key or os.getenv('GROQ_API_KEY', '')
        self.request_timeout = request_timeout

    @staticmethod
    def _format_reply(text):
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if len(lines) <= 1:
            sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z"])', text.strip())
            lines = [s for s in sentences if s]
        return '\n\n'.join(lines)

    def _groq_headers(self):
        return {
            'Authorization': f'Bearer {self.groq_api_key}',
            'Content-Type': 'application/json',
        }

    def _groq_payload(self, messages, tools):
        payload = {
            'model': MODEL_GROQ,
            'messages': messages,
            'temperature': float(os.getenv('GROQ_TEMPERATURE', '0.3')),
            'max_tokens': int(os.getenv('GROQ_MAX_TOKENS', '900')),
            'stream': False,
        }
        if tools:
            payload['tools'] = tools
            payload['tool_choice'] = 'auto'
        return payload

    def _post_groq(self, messages, tools):
        response = httpx.post(
            f'{BASE_URL_OPENAI}/chat/completions',
            headers=self._groq_headers(),
            json=self._groq_payload(messages, tools),
            timeout=self.request_timeout,
        )
        response.raise_for_status()
        return response.json()

    def _post_ollama(self, messages, tools):
        num_ctx = int(os.getenv('OLLAMA_NUM_CTX', '4096'))
        payload = {
            'model': MODEL_OLLAMA,
            'messages': messages,
            'stream': False,
            'options': {
                'temperature': 0.3,
                'num_ctx': num_ctx,
                'num_predict': int(os.getenv('OLLAMA_MAX_TOKENS', '900')),
            },
        }
        if tools:
            payload['tools'] = tools
        response = httpx.post(OLLAMA_API, json=payload, timeout=float(os.getenv('OLLAMA_TIMEOUT', '300')))
        response.raise_for_status()
        return response.json()

    @staticmethod
    def ollama_available():
        try:
            result = httpx.get(f'{OLLAMA_BASE}/api/tags', timeout=2.5)
            if result.status_code != 200:
                return False
            models = result.json().get('models', [])
            return any(MODEL_OLLAMA in (m.get('name') or '') for m in models)
        except Exception:
            return False

    def status(self):
        return {
            'groq_api_key_configured': bool(self.groq_api_key),
            'groq_model': MODEL_GROQ,
            'ollama_available': self.ollama_available(),
            'ollama_model': MODEL_OLLAMA,
            'ollama_url': OLLAMA_BASE,
            'tools': [t['function']['name'] for t in self.tools],
        }

    def chat(self, messages, provider='auto', context=None):
        system = self.system_prompt
        if context:
            system = f'{system}\n\n---\nCurrent page context provided by the app:\n{context}\n---'
        history = [{'role': 'system', 'content': system}] + messages

        attempts = []
        if provider in ('auto', 'groq') and self.groq_api_key:
            attempts.append(('groq', self._post_groq))
        if provider in ('auto', 'ollama'):
            attempts.append(('ollama', self._post_ollama))

        for name, poster in attempts:
            result = self._try_provider(name, poster, history)
            if result is not None:
                return result

        return {
            'reply': (
                'Millow AI is currently unavailable: Groq is not configured (set GROQ_API_KEY) '
                f'and the Ollama fallback at {OLLAMA_BASE} is offline. Start Ollama, or add an '
                'API key to ai/.env and restart the server.'
            ),
            'provider': None,
            'offline': True,
        }

    def _try_provider(self, name, poster, history):
        try:
            messages = [dict(m) for m in history]
            for _ in range(self.max_loops):
                data = poster(messages, self.tools)

                if name == 'groq':
                    message = data['choices'][0]['message']
                    content = message.get('content') or ''
                    tool_calls = message.get('tool_calls') or []
                else:
                    message = data.get('message', {})
                    content = message.get('content') or ''
                    tool_calls = message.get('tool_calls') or []

                if not tool_calls:
                    return {
                        'reply': self._format_reply(content),
                        'provider': name,
                        'model': f'{MODEL_GROQ if name == "groq" else MODEL_OLLAMA}',
                        'offline': False,
                    }

                messages.append({'role': 'assistant', 'content': content, 'tool_calls': tool_calls})

                for call in tool_calls:
                    call_id = call.get('id')
                    function = call.get('function', {})
                    fn_name = function.get('name', '')
                    arguments = function.get('arguments', {})
                    try:
                        if isinstance(arguments, str):
                            arguments = json.loads(arguments or '{}')
                        elif not isinstance(arguments, dict):
                            arguments = {}
                    except json.JSONDecodeError:
                        arguments = {}

                    handler = self.handlers.get(fn_name)
                    if handler is None:
                        tool_output = json.dumps({'error': f'unknown tool: {fn_name}'})
                    else:
                        try:
                            tool_output = json.dumps(handler(**arguments), default=str)
                        except Exception as error:
                            tool_output = json.dumps({'error': str(error)})

                    tool_message = {
                        'role': 'tool',
                        'content': tool_output,
                    }
                    if call_id:
                        tool_message['tool_call_id'] = call_id
                    messages.append(tool_message)

            return {
                'reply': 'I hit my processing limit. Try rephrasing the question.',
                'provider': name,
                'model': MODEL_GROQ if name == 'groq' else MODEL_OLLAMA,
                'offline': False,
            }
        except Exception as error:
            print(f'[chat] provider {name!r} failed: {error}')
            return None