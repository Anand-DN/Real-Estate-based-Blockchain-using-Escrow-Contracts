import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor

import httpx
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), '.env'))

BASE_URL_OPENAI = 'https://api.groq.com/openai/v1'
OLLAMA_BASE = os.getenv('OLLAMA_URL', 'http://localhost:11434')
OLLAMA_API = f'{OLLAMA_BASE}/api/chat'

# Groq retired llama-3.3-70b-versatile (it now answers 404), so the only
# general-purpose chat models left on Groq are the gpt-oss reasoning pair and
# qwen3.8.  gpt-oss-20b is the default no more: on the MREID risk question it
# retyped the Indian digit grouping ("₹93.69 Lakh" -> "₹9.369 Lakh",
# "₹1.02 Crore" -> "₹10.240 Lakh") and inverted the valuation gap, reporting the
# listing as 9.3% ABOVE the AI estimate when the record says 9.3% BELOW.
# qwen3.8-27b is not a reasoning model, so it does not spend its token budget
# before answering, and it reports those figures correctly.
MODEL_GROQ = os.getenv('GROQ_MODEL', 'qwen/qwen3.8-27b')
MODEL_OLLAMA = os.getenv('OLLAMA_MODEL', 'llama3.1:8b')

# Groq meters the free tier per model, so a 429 on the primary does not have to
# mean the whole key is spent for the minute: the alternates below are usually
# still inside their own quota and answer the turn.  Order is most-capable
# first.  gpt-oss-120b leads because 20b is the model that retyped Indian digit
# grouping and inverted the valuation gap (see the note on MODEL_GROQ); it is
# kept only as a last Groq resort before dropping to the local model.
GROQ_FALLBACK_MODELS = [
    m.strip() for m in os.getenv(
        'GROQ_FALLBACK_MODELS', 'openai/gpt-oss-120b,openai/gpt-oss-20b'
    ).split(',') if m.strip()
]

# The gpt-oss family spends its token budget on hidden reasoning and can return
# an empty completion once the cap is reached, so it needs the effort knob.  The
# effort itself is always defined, not only when the primary happens to be a
# reasoning model, because the fallback list can be one.
REASONING_MODELS = 'gpt-oss' in (MODEL_GROQ or '')
REASONING_EFFORT = os.getenv('GROQ_REASONING_EFFORT', 'low')
GROQ_TEMPERATURE = float(os.getenv('GROQ_TEMPERATURE', '0.2'))
# 600 comfortably fits the 60-150 word target the prompts set, and Groq bills
# per completion token, so a runaway generation is now both faster and cheaper.
GROQ_MAX_TOKENS = int(os.getenv('GROQ_MAX_TOKENS', '600'))
# One retry from a 1s base: a 5xx is usually transient, but a second wait only
# adds a second of dead time before failing over to a model that can answer now.
GROQ_BACKOFF = float(os.getenv('GROQ_BACKOFF_SECONDS', '1'))
GROQ_RETRIES = int(os.getenv('GROQ_RETRIES', '1'))

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

    def _groq_payload(self, messages, tools, model=None):
        active_model = model or MODEL_GROQ
        payload = {
            'model': active_model,
            'messages': messages,
            'temperature': GROQ_TEMPERATURE,
            'max_tokens': GROQ_MAX_TOKENS,
            'stream': False,
        }

        # gpt-oss-20b is a reasoning model and burns most of its budget on
        # hidden reasoning: a single tool-calling turn spent 483 of 517
        # completion tokens thinking and returned no content at all, which the
        # agent then rendered as an empty bubble. "low" keeps the reasoning
        # short so the answer actually fits in the budget.  Decided per model,
        # not off the primary: the gpt-oss alternates are reasoning models even
        # when the primary is qwen, and without the knob they return empty
        # completions that surface as a blank bubble instead of an answer.
        if 'gpt-oss' in (active_model or '') and REASONING_EFFORT:
            payload['reasoning_effort'] = REASONING_EFFORT
        if tools:
            payload['tools'] = tools
            payload['tool_choice'] = 'auto'
        return payload

    def _post_groq(self, messages, tools, model=None):
        payload = self._groq_payload(messages, tools, model)
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
                # A 429 is a per-minute quota window. It cannot clear in the two
                # to six seconds this backoff would wait, so retrying only turned
                # a fast, honest "rate limited" reply into a ~13s one. Fail now
                # and let the caller show the accurate message.
                if response.status_code == 429:
                    raise httpx.HTTPStatusError(
                        f'429 Too Many Requests: {response.text[:200]}',
                        request=response.request,
                        response=response,
                    )
                if response.status_code < 500:
                    response.raise_for_status()
                    return response.json()
                # 5xx is genuinely transient, so a retry can genuinely help.
                last_error = httpx.HTTPStatusError(
                    f'{response.status_code} from Groq: {response.text[:200]}',
                    request=response.request,
                    response=response,
                )
            if attempt < GROQ_RETRIES:
                time.sleep(GROQ_BACKOFF * (2 ** attempt))
        raise last_error

    def _post_ollama(self, messages, tools):
        num_ctx = int(os.getenv('OLLAMA_NUM_CTX', '2048'))
        payload = {
            'model': self.resolve_ollama_model() or MODEL_OLLAMA,
            'messages': messages,
            'stream': False,
            # Keeps the model resident in RAM between turns.  Ollama otherwise
            # unloads it after 5 minutes of idle, and the next request pays a
            # 6-8s cold load before it generates a single token - measured on
            # this i7-1265U as roughly 12s cold vs 4s warm for the same reply.
            'keep_alive': os.getenv('OLLAMA_KEEP_ALIVE', '1h'),
            'options': {
                'temperature': 0.2,
                'num_ctx': num_ctx,
                'num_predict': int(os.getenv('OLLAMA_MAX_TOKENS', '600')),
                'num_thread': int(os.getenv('OLLAMA_NUM_THREAD', '4')),
            },
        }
        if tools:
            payload['tools'] = tools
        response = httpx.post(OLLAMA_API, json=payload, timeout=float(os.getenv('OLLAMA_TIMEOUT', '300')))
        response.raise_for_status()
        return response.json()

    _ollama_cache = {'model': None, 'checked_at': 0.0}
    OLLAMA_CACHE_SECONDS = float(os.getenv('OLLAMA_PROBE_CACHE_SECONDS', '30'))

    @staticmethod
    def ollama_models():
        try:
            result = httpx.get(f'{OLLAMA_BASE}/api/tags', timeout=2.5)
            if result.status_code != 200:
                return []
            return [(m.get('name') or '') for m in result.json().get('models', [])]
        except Exception:
            return []

    @classmethod
    def resolve_ollama_model(cls):
        """The tag to actually ask for: the configured one if it is pulled,
        otherwise any installed llama.  Matching on the exact tag meant a machine
        with llama3.2:latest pulled instead of llama3.2:3b reported the local
        fallback as offline, so a 429 from Groq ended the turn with the
        rate-limit message even though a perfectly good model was sitting there.

        The probe result is cached briefly: this runs on the fallback path of a
        throttled turn, and a 2.5s /api/tags round trip on top of an already slow
        local generation is pure added latency for an answer that cannot change
        from one second to the next.
        """
        now = time.monotonic()
        cached = cls._ollama_cache
        if cached['model'] and (now - cached['checked_at']) < cls.OLLAMA_CACHE_SECONDS:
            return cached['model']

        installed = cls.ollama_models()
        chosen = None
        for name in installed:
            if name == MODEL_OLLAMA or name.split(':')[0] == MODEL_OLLAMA.split(':')[0]:
                chosen = name
                break
        if chosen is None:
            for name in installed:
                if 'llama' in name.lower():
                    chosen = name
                    break
        # Only a positive result is cached: a negative one is usually a server
        # that is still starting, and caching that would keep reporting offline.
        if chosen:
            cls._ollama_cache = {'model': chosen, 'checked_at': now}
        return chosen

    @classmethod
    def ollama_available(cls):
        return cls.resolve_ollama_model() is not None

    def status(self):
        return {
            'groq_api_key_configured': bool(self.groq_api_key),
            'groq_model': MODEL_GROQ,
            'groq_fallback_models': GROQ_FALLBACK_MODELS,
            'ollama_available': self.ollama_available(),
            'ollama_model': MODEL_OLLAMA,
            'ollama_url': OLLAMA_BASE,
            'tools': [t['function']['name'] for t in self.tools],
            'reasoning_model': REASONING_MODELS,
            'reasoning_effort': REASONING_EFFORT or None,
            'max_tokens': GROQ_MAX_TOKENS,
        }

    def chat(self, messages, provider='auto', context=None, tools=None):
        system = self.system_prompt
        if context:
            system = f'{system}\n\n---\nCurrent page context provided by the app:\n{context}\n---'
        history = [{'role': 'system', 'content': system}] + messages
        # Per-request tool subset (see mreid_tools.select_tools).  Falls back to
        # the agent's full catalogue when the caller does not narrow it.
        active_tools = self.tools if tools is None else tools

        # Every attempt is (provider label, poster, model).  The label stays
        # 'groq' for the alternates so callers and the degraded flag still see
        # one provider, while the model is carried through to the response.
        attempts = []
        if provider in ('auto', 'groq') and self.groq_api_key:
            groq_models = [MODEL_GROQ] + [m for m in GROQ_FALLBACK_MODELS if m != MODEL_GROQ]
            for model in groq_models:
                attempts.append((
                    'groq',
                    lambda msgs, tls, _m=model: self._post_groq(msgs, tls, _m),
                    model,
                ))
        if provider in ('auto', 'ollama'):
            attempts.append(('ollama', self._post_ollama, MODEL_OLLAMA))

        failures = []
        for name, poster, model in attempts:
            result = self._try_provider(name, poster, history, active_tools, model=model)
            if result is not None:
                # The local model answers correctly but takes two minutes, so
                # say so instead of letting it look like a hang.
                if name != 'groq' and failures:
                    result['degraded'] = True
                    result['degraded_reason'] = failures[-1]
                elif name == 'groq' and model != MODEL_GROQ and failures:
                    # Answered, but not by the primary model, so the user should
                    # know the accuracy note that came with it.
                    result['degraded'] = True
                    result['degraded_reason'] = failures[-1]
                return result
            failures.append(f'{name} ({model}): {self._last_error or "unknown error"}')

        # A rate limit is not a configuration problem, and telling a user to set
        # an API key they already set sends them off to debug the wrong thing.
        # Name the actual cause.
        throttled = [f for f in failures if '429' in f or 'Rate limit' in f]
        if throttled and self.groq_api_key:
            reply = (
                "Millow AI hit Groq's rate limit just now on every model it has, so it "
                'could not answer. This is a usage cap on the free key, not a '
                'misconfiguration. Wait about a minute and ask again, or start the local '
                f'fallback with: ollama pull {MODEL_OLLAMA}'
            )
        elif self.groq_api_key:
            reply = (
                'Millow AI could not reach Groq, and the Ollama fallback at '
                f'{OLLAMA_BASE} is offline. Check the network, or start Ollama.'
            )
        else:
            reply = (
                'Millow AI is currently unavailable: Groq is not configured (set GROQ_API_KEY) '
                f'and the Ollama fallback at {OLLAMA_BASE} is offline. Start Ollama, or add an '
                'API key to ai/.env and restart the server.'
            )

        return {
            'reply': reply,
            'provider': None,
            'offline': True,
            'degraded': True,
            'throttled': bool(throttled),
            'degraded_reason': '; '.join(failures) or 'no provider was reachable',
        }

    def _run_tool_calls(self, tool_calls):
        """Execute one assistant turn's tool calls, in order, concurrently.

        A single turn can ask for several tools at once (the MREID risk question
        triggers get_mreid_risk and get_mreid_property together).  Running them
        in a loop serialised the backend round trips, so the turn waited for
        each one in turn.  Results are returned in the original order because
        the `tool` messages have to line up with `tool_calls`.
        """
        def run(call):
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

            tool_message = {'role': 'tool', 'content': tool_output}
            if call_id:
                tool_message['tool_call_id'] = call_id
            return tool_message

        if len(tool_calls) == 1:
            return [run(tool_calls[0])]
        with ThreadPoolExecutor(max_workers=len(tool_calls)) as pool:
            return [f.result() for f in [pool.submit(run, c) for c in tool_calls]]

    def _try_provider(self, name, poster, history, tools=None, model=None):
        active_tools = self.tools if tools is None else tools
        active_model = model or (MODEL_GROQ if name == 'groq' else MODEL_OLLAMA)
        try:
            messages = [dict(m) for m in history]
            for _ in range(self.max_loops):
                data = poster(messages, active_tools)

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
                        'model': active_model,
                        'offline': False,
                    }

                messages.append({'role': 'assistant', 'content': content, 'tool_calls': tool_calls})

                for tool_message in self._run_tool_calls(tool_calls):
                    messages.append(tool_message)

            return {
                'reply': 'I hit my processing limit. Try rephrasing the question.',
                'provider': name,
                'model': active_model,
                'offline': False,
            }
        except Exception as error:
            self._last_error = f'{type(error).__name__}: {error}'
            print(f'[chat] provider {name!r} failed: {self._last_error}')
            return None