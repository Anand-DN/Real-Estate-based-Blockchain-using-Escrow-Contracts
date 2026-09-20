import { useEffect, useRef, useState } from "react";

const AI_BASE = process.env.REACT_APP_AI_URL || "http://localhost:8000";
const STORAGE_KEY = "millow_chat_history";

const WELCOME = {
  role: "assistant",
  content:
    "Hi, I'm Millow AI. Ask me about valuations, risk on a listing, fraud checks, or the best deals on the market.",
};

const loadHistory = () => {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    const parsed = raw ? JSON.parse(raw) : null;
    if (Array.isArray(parsed) && parsed.length) return parsed;
  } catch {
    /* ignore corrupt storage */
  }
  return [WELCOME];
};

const ChatBot = ({ property, realEstate, homes, account }) => {
  const [open, setOpen] = useState(false);
  const [messages, setMessages] = useState(loadHistory);
  const [draft, setDraft] = useState("");
  const [busy, setBusy] = useState(false);
  const [provider, setProvider] = useState(null);
  const [unread, setUnread] = useState(true);
  const [owners, setOwners] = useState([]);
  const bodyRef = useRef(null);
  const inputRef = useRef(null);

  const propertyId = property?.tokenId ?? property?.id;
  const propertyName = property?.name || null;

  const SUGGESTIONS = [
    {
      id: "price",
      label: "Price",
      icon: (
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
          <path d="M20.59 13.41l-7.17 7.17a2 2 0 0 1-2.83 0L2 12V2h10l8.59 8.59a2 2 0 0 1 0 2.82z" />
          <line x1="7" y1="7" x2="7.01" y2="7" />
        </svg>
      ),
      prompt: propertyName
        ? `Predict the price of ${propertyName}.`
        : propertyId
          ? `Predict the price of property with token id ${propertyId}.`
          : "How does Millow predict a property's price?",
    },
    {
      id: "risk",
      label: "Risk",
      icon: (
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
          <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z" />
        </svg>
      ),
      prompt: propertyName
        ? `What is the risk level for ${propertyName}?`
        : propertyId
          ? `What is the risk level for property with token id ${propertyId}?`
          : "How does Millow calculate transaction risk?",
    },
    {
      id: "details",
      label: "Details",
      icon: (
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
          <circle cx="12" cy="12" r="10" />
          <line x1="12" y1="16" x2="12" y2="12" />
          <line x1="12" y1="8" x2="12.01" y2="8" />
        </svg>
      ),
      prompt: propertyName
        ? `Give me the AI insights for ${propertyName}.`
        : propertyId
          ? `Give me the AI insights for property with token id ${propertyId}.`
          : "What insights does Millow provide for a property?",
    },
    {
      id: "buy",
      label: "How to buy",
      icon: (
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
          <circle cx="12" cy="12" r="10" />
          <path d="M9.09 9a3 3 0 0 1 5.83 1c0 2-3 3-3 3" />
          <line x1="12" y1="17" x2="12.01" y2="17" />
        </svg>
      ),
      prompt: "How do I buy a property on Millow?",
    },
  ];

  useEffect(() => {
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(messages.slice(-40)));
    } catch {
      /* storage full or unavailable */
    }
  }, [messages]);

  useEffect(() => {
    if (open && bodyRef.current) {
      bodyRef.current.scrollTop = bodyRef.current.scrollHeight;
    }
  }, [messages, busy, open]);

  // Auto-grow the textarea so multi-line messages (Shift+Enter) stay readable
  // instead of scrolling inside a one-line box.
  useEffect(() => {
    const el = inputRef.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = `${Math.min(el.scrollHeight, 120)}px`;
  }, [draft, open, messages]);

  const toggle = () => {
    setOpen((prev) => {
      if (!prev) setUnread(false);
      return !prev;
    });
    if (!open) setTimeout(() => inputRef.current && inputRef.current.focus(), 60);
  };

  useEffect(() => {
    const closePanel = () => setOpen(false);
    window.addEventListener("millow:close-chat", closePanel);
    return () => window.removeEventListener("millow:close-chat", closePanel);
  }, []);

  // Resolve the current on-chain owner of every listing so Millow can answer
  // "who owns X?" from live contract data (ownerOf) instead of guessing.
  // Fetched in parallel only when the chat is opened, so we don't hit the
  // chain on every page load.
  useEffect(() => {
    if (!open || !realEstate || !homes.length) return;
    let cancelled = false;
    Promise.all(
      homes.map(async (home) => {
        try {
          const owner = await realEstate.ownerOf(home.tokenId);
          return [home.name, home.tokenId, owner];
        } catch {
          /* token not found or RPC error: skip it */
          return null;
        }
      }),
    ).then((rows) => {
      if (!cancelled) setOwners(rows.filter(Boolean));
    });
    return () => {
      cancelled = true;
    };
  }, [open, realEstate, homes]);

  const send = async (override) => {
    const text = (override ?? draft).trim();
    if (!text || busy) return;
    if (!override) setDraft("");
    setBusy(true);

    const userMessage = { role: "user", content: text };
    const nextMessages = [...messages, userMessage];
    setMessages(nextMessages);

    // Tell the AI which property is on screen so it refers to it by name,
    // not as "Property 3".
    const parts = [];
    if (propertyName) {
      parts.push(
        `The user is currently viewing the property "${propertyName}" (token id ${propertyId}). When answering their questions, always refer to this property by its name "${propertyName}".`,
      );
    }
    if (owners.length) {
      const list = owners
        .map(([name, tokenId, owner]) => `- ${name} (token id ${tokenId}): ${owner}`)
        .join("\n");
      parts.push(
        `Known current property owners, read live from the blockchain with ownerOf:\n${list}\n` +
          `If the user asks who owns a property, answer with its wallet address from this list. ` +
          `Never invent an owner: if the property is not in this list, say the owner could not be determined.`,
      );
    }
    const context = parts.length ? parts.join("\n\n") : null;

    try {
      const response = await fetch(`${AI_BASE}/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ messages: nextMessages, context }),
      });
      const data = await response.json();
      const reply = data.reply?.trim() || "I didn't get that. Could you rephrase?";
      setMessages([
        ...nextMessages,
        { role: "assistant", content: reply },
      ]);
      setProvider(data.provider || null);
    } catch {
      setMessages([
        ...nextMessages,
        {
          role: "assistant",
          content:
            "I couldn't reach the Millow AI server. Make sure it's running with `npm run ai`.",
        },
      ]);
      setProvider("offline");
    } finally {
      setBusy(false);
    }
  };

  const onKeyDown = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      send();
    }
  };

  const reset = () => {
    setMessages([WELCOME]);
    setProvider(null);
  };

  return (
    <div className="chatbot">
      {open && (
        <div className="chatbot__panel">
          <div className="chatbot__header">
            <div className="chatbot__title">
              <span className="chatbot__avatar">AI</span>
              <div>
                <strong>Millow AI</strong>
                <span className="chatbot__status">
                  {busy
                    ? "Thinking..."
                    : provider
                      ? `via ${provider}`
                      : "Online"}
                </span>
              </div>
            </div>
            <button type="button" className="chatbot__reset" title="Clear chat" onClick={reset}>
              Clear
            </button>
          </div>

          <div className="chatbot__body" ref={bodyRef}>
            {messages.map((message, index) => (
              <div
                key={index}
                className={`chatbot__msg chatbot__msg--${message.role}`}
              >
                {message.content}
              </div>
            ))}
            {busy && (
              <div className="chatbot__typing" role="status" aria-label="Millow is typing">
                <span className="chatbot__dot" />
                <span className="chatbot__dot" />
                <span className="chatbot__dot" />
              </div>
            )}
          </div>

          <div className="chatbot__chips">
            {SUGGESTIONS.map((suggestion) => (
              <button
                key={suggestion.id}
                type="button"
                className="chatbot__chip"
                disabled={busy}
                onClick={() => send(suggestion.prompt)}
              >
                {suggestion.icon}
                {suggestion.label}
              </button>
            ))}
          </div>

          <div className="chatbot__footer">
            <textarea
              ref={inputRef}
              className="chatbot__input"
              rows="1"
              placeholder="Ask about pricing, risk, fraud..."
              value={draft}
              onChange={(e) => setDraft(e.target.value)}
              onKeyDown={onKeyDown}
              disabled={busy}
            />
            <button
              type="button"
              className="chatbot__send"
              onClick={send}
              disabled={busy || !draft.trim()}
            >
              Send
            </button>
          </div>
        </div>
      )}

      <button type="button" className="chatbot__fab" onClick={toggle} title="Millow AI assistant">
        {open ? (
          <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.4" strokeLinecap="round">
            <path d="M18 6 6 18M6 6l12 12" />
          </svg>
        ) : (
          <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <path d="M3 10.5 12 3l9 7.5" />
            <path d="M5 9.5V21h14V9.5" />
            <path d="M9 21v-6h6v6" />
          </svg>
        )}
        {!open && unread && <span className="chatbot__badge" />}
      </button>
    </div>
  );
};

export default ChatBot;