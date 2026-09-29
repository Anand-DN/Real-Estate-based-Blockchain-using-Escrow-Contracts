import json
import os
import re
import time

import httpx
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), '.env'))

BASE_URL_OPENAI = 'https://api.groq.com/openai/v1'
OLLAMA_BASE = os.getenv('OLLAMA_URL', 'http://localhost:11434')
OLLAMA_API = f'{OLLAMA_BASE}/api/chat'

MODEL_GROQ = os.getenv('GROQ_MODEL', 'llama-3.3-70b-versatile')
MODEL_OLLAMA = os.getenv('OLLAMA_MODEL', 'llama3.1:8b')

# The gpt-oss family spends its token budget on hidden reasoning and can return
# an empty completion once the cap is reached, so it needs the effort knob.
REASONING_MODELS = 'gpt-oss' in (MODEL_GROQ or '')
REASONING_EFFORT = os.getenv('GROQ_REASONING_EFFORT', 'low') if REASONING_MODELS else ''

# A 429 is a quota window rather than a bad request, so it is worth waiting out
# rather than failing straight over to the local model.
# The free Groq tier allows ~8k tokens per minute for the whole org, and this
# app is the only caller, so a chat turn is easily half the budget. A 429 needs
# the quota window to refill, which takes up to a minute, so the backoff is
# sized to cover it rather than to fail fast onto the 2-minute local model.
GROQ_RETRIES = int(os.getenv('GROQ_RETRIES', '3'))
GROQ_BACKOFF = float(os.getenv('GROQ_BACKOFF_SECONDS', '8'))

# Shown instead of an empty bubble when a provider returns no text at all.
EMPTY_REPLY = (
    'Millow AI could not produce an answer for that. Please try rephrasing, '
    'or ask about a specific property, price or risk detail.'
)

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
    '- Lead with the answer. The first sentence must state the number, price or verdict the user '
    'asked for. No preamble, no restating the question, no closing summary of what you just said.\n'
    '- Be brief. Target 60-120 words total unless the user asked for depth. Prefer 2-4 short '
    'sentences over an exhaustive breakdown.\n'
    '- Only break into a bulleted list when there are 3 or more genuinely separate items. Never pad '
    'a simple answer into a list.\n'
    '- Never produce a pipe table or a markdown heading. The app renders your text as-is.\n'
    '- Write "49.7%" and "₹82.05 Lakh" with no space between the symbol and the number.\n'
    '- Use the tools, but do not restate the raw tool output. Report the conclusion, not the dump.\n'
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
        self._last_error = None

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
            'max_tokens': int(os.getenv('GROQ_MAX_TOKENS', '1400')),
            'stream': False,
        }
        # gpt-oss-20b is a reasoning model and burns most of its budget on
        # hidden reasoning: a single tool-calling turn spent 483 of 517
        # completion tokens thinking and returned no content at all, which the
        # agent then rendered as an empty bubble. "low" keeps the reasoning
        # short so the answer actually fits in the budget.
        if REASONING_MODELS and REASONING_EFFORT:
            payload['reasoning_effort'] = REASONING_EFFORT
        if tools:
            payload['tools'] = tools
            payload['tool_choice'] = 'auto'
        return payload

    def _post_groq(self, messages, tools):
        payload = self._groq_payload(messages, tools)
        last_error = None
        for attempt in range(GROQ_RETRIES + 1):
            try:
                response = httpx.post(
                    f'{BASE_URL_OPENAI}/chat/completions',
                    headers=self._groq_headers(),
                    json=payload,
                    timeout=self.request_timeout,
                )
            except httpx.HTTPError as error:
                last_error = error
            else:
                if response.status_code != 429:
                    response.raise_for_status()
                    return response.json()
                last_error = httpx.HTTPStatusError(
                    f'429 Too Many Requests: {response.text[:200]}',
                    request=response.request,
                    response=response,
                )
            # 429 is a quota window, not a bad request, so waiting and retrying
            # is worth it. Backing off here keeps a burst of chat turns from
            # dumping the user straight onto the slow local fallback.
            if attempt < GROQ_RETRIES:
                time.sleep(GROQ_BACKOFF * (2 ** attempt))
        raise last_error

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
            'reasoning_model': REASONING_MODELS,
            'reasoning_effort': REASONING_EFFORT or None,
            'max_tokens': int(os.getenv('GROQ_MAX_TOKENS', '1400')),
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

        failures = []
        for name, poster in attempts:
            result = self._try_provider(name, poster, history)
            if result is not None:
                # The local model answers correctly but takes two minutes, so
                # say so instead of letting it look like a hang.
                if name != 'groq' and failures:
                    result['degraded'] = True
                    result['degraded_reason'] = failures[-1]
                return result
            failures.append(f'{name}: {self._last_error or "unknown error"}')

        return {
            'reply': (
                'Millow AI is currently unavailable: Groq is not configured (set GROQ_API_KEY) '
                f'and the Ollama fallback at {OLLAMA_BASE} is offline. Start Ollama, or add an '
                'API key to ai/.env and restart the server.'
            ),
            'provider': None,
            'offline': True,
            'degraded': True,
            'degraded_reason': '; '.join(failures) or 'no provider was reachable',
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
                    # A reasoning model that exhausted its budget on internal
                    # reasoning returns content="" and no tool calls. Treating
                    # that as a finished answer is what produced empty bubbles,
                    # so spend the last loop re-asking for prose only.
                    if not content.strip() and name == 'groq' and _ < self.max_loops - 1:
                        print('[chat] groq returned an empty completion; retrying for text')
                        messages.append({'role': 'user', 'content': 'Reply with the answer only.'})
                        continue
                    return {
                        'reply': self._format_reply(content) or EMPTY_REPLY,
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
            self._last_error = f'{type(error).__name__}: {error}'
            print(f'[chat] provider {name!r} failed: {self._last_error}')
            return None