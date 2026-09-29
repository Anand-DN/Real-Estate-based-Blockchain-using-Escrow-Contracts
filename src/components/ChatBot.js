import { useEffect, useRef, useState } from "react";
import { propertyById, propertyChain } from "../lib/millowApi";

const AI_BASE = process.env.REACT_APP_AI_URL || "http://localhost:8000";
const STORAGE_KEY = "millow_chat_history_mreid";
const ENDPOINT = "/chat/mreid";

const WELCOME = {
  role: "assistant",
  content:
    "Hi, I'm MILLOW AI. I can help you find properties, explain what the AI estimate means, put a price in context with the market, and explain how a purchase, escrow or listing works on chain. Ask me about any listing, or name an MREID like MREID_0000017.",
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

const Icon = {
  search: (
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <circle cx="11" cy="11" r="7" />
      <line x1="21" y1="21" x2="16.5" y2="16.5" />
    </svg>
  ),
  tag: (
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M20.59 13.41l-7.17 7.17a2 2 0 0 1-2.83 0L2 12V2h10l8.59 8.59a2 2 0 0 1 0 2.82z" />
      <line x1="7" y1="7" x2="7.01" y2="7" />
    </svg>
  ),
  chart: (
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M3 3v18h18" />
      <path d="m7 14 4-4 3 3 5-6" />
    </svg>
  ),
  info: (
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <circle cx="12" cy="12" r="10" />
      <line x1="12" y1="16" x2="12" y2="12" />
      <line x1="12" y1="8" x2="12.01" y2="8" />
    </svg>
  ),
  alert: (
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M10.3 3.9 1.8 18a2 2 0 0 0 1.7 3h17a2 2 0 0 0 1.7-3L13.7 3.9a2 2 0 0 0-3.4 0z" />
      <line x1="12" y1="9" x2="12" y2="13" />
      <line x1="12" y1="17" x2="12.01" y2="17" />
    </svg>
  ),
  link: (
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M10 13a5 5 0 0 0 7.5.5l3-3a5 5 0 0 0-7-7l-1.7 1.7" />
      <path d="M14 11a5 5 0 0 0-7.5-.5l-3 3a5 5 0 0 0 7 7l1.7-1.7" />
    </svg>
  ),
  trend: (
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <polyline points="22 7 13.5 15.5 8.5 10.5 2 17" />
      <polyline points="16 7 22 7 22 13" />
    </svg>
  ),
};

// Quick actions follow the brief: help users find properties, understand
// valuation, read the market, and follow a transaction.  With a property open
// they become the questions a buyer actually asks about that listing: what it
// costs, what is risky about it, what exactly is on offer, and what the
// location is like.
const actionsFor = (context) => {
  const name = context ? `${context.location}, ${context.city}` : null;
  if (context) {
    return [
      {
        id: "price",
        label: "Price",
        icon: Icon.tag,
        prompt: `What is the price of ${name} (${context.mreid_id})? Give the listed price, the price per sqft, and the MILLOW AI estimate, and say whether the listing is above or below it.`,
      },
      {
        id: "risk",
        label: "Risk",
        icon: Icon.alert,
        prompt: `What are the risks of buying ${name} (${context.mreid_id})? Run your risk analysis and explain the score, the anomaly check and anything a buyer should verify before paying.`,
      },
      {
        id: "details",
        label: "Details",
        icon: Icon.info,
        prompt: `Give me the full details of ${name} (${context.mreid_id}): configuration, area, floor, facing, age, amenities and any other catalogue facts on file.`,
      },
      {
        id: "locality",
        label: "Locality",
        icon: Icon.chart,
        prompt: `How does ${name} compare with other properties in ${context.location}, ${context.city}? Give the median price per sqft and what the locality is like to live in.`,
      },
      {
        id: "chain",
        label: "On-chain",
        icon: Icon.link,
        prompt: `What is the blockchain status of ${name} (${context.mreid_id}) — is it tokenized, who holds the NFT, is it listed for sale, and is there an active escrow sale?`,
      },
      {
        id: "forecast",
        label: "Forecast",
        icon: Icon.trend,
        prompt: `What will the price of ${name} (${context.mreid_id}) be in the next 15 years? Project the value forward and say what you assumed.`,
      },
      {
        id: "similar",
        label: "Similar",
        icon: Icon.search,
        prompt: `Find properties similar to ${name} that are listed below their AI estimate.`,
      },
    ];
  }
  return [
    {
      id: "find",
      label: "Find properties",
      icon: Icon.search,
      prompt:
        "Find 3 potentially undervalued 2-bedroom apartments in Mumbai under 1.2 crore.",
    },
    {
      id: "valuation",
      label: "Explain AI valuation",
      icon: Icon.tag,
      prompt:
        "How does the MILLOW AI estimate work, and how should I read it next to a listed price?",
    },
    {
      id: "market",
      label: "Market trends",
      icon: Icon.chart,
      prompt:
        "Give me the key market insights for the MREID catalogue, city by city.",
    },
    {
      id: "transaction",
      label: "How buying works",
      icon: Icon.info,
      prompt:
        "How does buying a property on MILLOW work, from escrow to the transfer of the property NFT?",
    },
  ];
};

// The assistant answers in Markdown, and the raw string used to be dropped
// straight into the bubble, so the **bold** markers showed up as literal text
// and every block ran together.  The reply is turned into real elements here:
// **text** becomes <strong>, "- " lines become a list, and blocks are separated
// by CSS rather than by literal blank lines.  Keys are derived from the line
// index, which is stable for a given message.
const BOLD = /\*\*(.+?)\*\*/g;
const UNORDERED = /^[-*•]\s+/;
const ORDERED = /^\d+[.)]\s+/;
const HEADING = /^#{1,6}\s+/;
const TABLE_RULE = /^\|?[\s:|-]+\|[\s:|-]*$/;
// The model pads prices and percentages with spaces around the symbol
// ("49.7 %", "₹ 82.05 Lakh").  Tightened so the figures read as one token.
const LOOSE_NUMBER = /\s+([%₹])|([%₹])\s+/g;

// Splits one line into text runs and <strong> runs, leaving the asterisks out.
const inline = (text, key) => {
  const parts = [];
  let last = 0;
  let match;
  BOLD.lastIndex = 0;
  while ((match = BOLD.exec(text)) !== null) {
    if (match.index > last) parts.push(text.slice(last, match.index));
    parts.push(<strong key={`${key}-${match.index}`}>{match[1]}</strong>);
    last = match.index + match[0].length;
  }
  if (last < text.length) parts.push(text.slice(last));
  return parts;
};

// Collapses the whitespace the model sprinkles around currency and percent
// signs, so a figure reads as one token instead of "₹ 82.05 Lakh".
const tidyNumbers = (text) => {
  let out = String(text).replace(/\s+%/g, "%");
  let previous;
  do {
    previous = out;
    out = out.replace(LOOSE_NUMBER, "$1$2");
  } while (out !== previous);
  return out;
};

const cells = (line) =>
  line
    .trim()
    .replace(/^\|/, "")
    .replace(/\|$/, "")
    .split("|")
    .map((c) => tidyNumbers(c.trim()));

const renderMessage = (text) => {
  const blocks = [];
  let items = null;
  let ordered = false;
  let rows = null;

  const flushList = () => {
    if (!items) return;
    const Tag = ordered ? "ol" : "ul";
    blocks.push(
      <Tag className="chatbot__msg-list" key={`list-${blocks.length}`}>
        {items}
      </Tag>,
    );
    items = null;
  };

  const flushTable = () => {
    if (!rows) return;
    const [head, ...body] = rows;
    blocks.push(
      <table className="chatbot__msg-table" key={`table-${blocks.length}`}>
        <thead>
          <tr>
            {head.map((c, i) => (
              <th key={i}>{inline(tidyNumbers(c), `th-${i}`)}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {body.map((r, i) => (
            <tr key={i}>
              {r.map((c, j) => (
                <td key={j}>{inline(tidyNumbers(c), `td-${i}-${j}`)}</td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>,
    );
    rows = null;
  };

  const flushAll = () => {
    flushList();
    flushTable();
  };

  String(text ?? "")
    .split("\n")
    .forEach((raw, i) => {
      const line = raw.trim();
      // A blank line separates blocks; it must not break a list in half, since
      // the model puts one blank line between bullets.
      if (!line) return;

      // Pipe table.  The |---|---| separator row is dropped, not rendered.
      if (line.startsWith("|")) {
        flushList();
        if (TABLE_RULE.test(line)) return;
        if (!rows) rows = [];
        rows.push(cells(line));
        return;
      }
      flushTable();

      if (UNORDERED.test(line) || ORDERED.test(line)) {
        const nowOrdered = ORDERED.test(line);
        // A bullet list directly after a numbered one (or the reverse) starts a
        // new list rather than silently changing the tag mid-run.
        if (items && ordered !== nowOrdered) flushList();
        if (!items) {
          items = [];
          ordered = nowOrdered;
        }
        const body = tidyNumbers(line.replace(nowOrdered ? ORDERED : UNORDERED, ""));
        items.push(<li key={i}>{inline(body, `li-${i}`)}</li>);
        return;
      }
      flushList();

      if (HEADING.test(line)) {
        blocks.push(
          <p className="chatbot__msg-h" key={`h-${i}`}>
            {inline(tidyNumbers(line.replace(HEADING, "")), `h-${i}`)}
          </p>,
        );
        return;
      }

      const parts = inline(tidyNumbers(line), `p-${i}`);
      // A line that is nothing but bold is a heading too: the model uses that
      // shape for "Listed price" / "Risk" style section labels.
      const isHeading = parts.length === 1 && typeof parts[0] !== "string";
      blocks.push(
        <p
          className={isHeading ? "chatbot__msg-h" : "chatbot__msg-p"}
          key={`p-${i}`}
        >
          {parts}
        </p>,
      );
    });

  flushAll();
  return blocks;
};

// The assistant is told exactly what the user is looking at, with the real
// numbers the property page is showing, and is told to use them instead of
// guessing.  The MREID agent still verifies everything through its own tools.
const buildContext = (detail, chain) => {
  if (!detail) return null;
  const ai = detail.ai_estimation || {};
  const property = detail.property || {};
  const signal = (detail.ai_market_signal || {}).label;
  const locality = detail.locality || {};
  const chainState = (chain && chain.chain) || {};

  const lines = [
    `The user is looking at ${property.location}, ${property.city} (${detail.mreid_id}).`,
    `Refer to it as "${property.location}" and answer questions about it in these terms.`,
    `Catalogue facts: ${property.bedrooms} bedrooms, ${property.area} sqft, listed price ${detail.listed_price_formatted} (${detail.price_per_sqft} per sqft).`,
    `MILLOW V5 research estimate: ${ai.ai_estimated_price_formatted} (${ai.ai_estimated_price_per_sqft} per sqft). Market signal: ${signal}.`,
    locality.median_price_per_sqft
      ? `Locality context: median ${locality.median_price_per_sqft} per sqft across ${locality.count} catalogue records in ${property.location}.`
      : null,
    chainState.token_id
      ? `Blockchain: token #${chainState.token_id}${chainState.owner ? `, held by ${chainState.owner}` : ""}, listed=${Boolean(
          chainState.listed,
        )}, active sale=${Boolean(chainState.active_sale)}.`
      : `Blockchain: tokenization state not reported for this property.`,
    "Use these numbers, and your own tools, for any question about this property. Never invent a price, area, owner or status, and say so when something is not available. Report prices in Indian Rupees.",
  ];

  return {
    text: lines.filter(Boolean).join("\n"),
    label: `${property.location}, ${property.city}`,
    location: property.location,
    city: property.city,
    mreid_id: detail.mreid_id,
  };
};

const ChatBot = ({ open = false, onOpenChange, propertyId = null }) => {
  const [messages, setMessages] = useState(loadHistory);
  const [draft, setDraft] = useState("");
  const [busy, setBusy] = useState(false);
  const [provider, setProvider] = useState(null);
  const [degraded, setDegraded] = useState(false);
  const [unread, setUnread] = useState(true);
  const [context, setContext] = useState(null);
  const bodyRef = useRef(null);
  const inputRef = useRef(null);
  const lastMsgRef = useRef(null);
  // Set when a reply lands so the effect below can reveal the TOP of that
  // reply.  Clearing it on send keeps the "typing" bubble in view.
  const revealTopRef = useRef(false);

  useEffect(() => {
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(messages.slice(-40)));
    } catch {
      /* storage full or unavailable */
    }
  }, [messages]);

  useEffect(() => {
    const body = bodyRef.current;
    if (!open || !body) return;

    if (revealTopRef.current && lastMsgRef.current) {
      // A long reply used to scroll its own first line off the top of the
      // panel, so the answer only ever showed its tail.  Anchor on the top of
      // the newest bubble instead of jumping to the bottom of the log.
      revealTopRef.current = false;
      const top = lastMsgRef.current.offsetTop - body.offsetTop;
      body.scrollTop = Math.max(0, top - 8);
    } else {
      body.scrollTop = body.scrollHeight;
    }
    setUnread(false);
  }, [messages, busy, open]);

  // Auto-grow the textarea so multi-line messages (Shift+Enter) stay readable
  // instead of scrolling inside a one-line box.
  useEffect(() => {
    const el = inputRef.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = `${Math.min(el.scrollHeight, 120)}px`;
  }, [draft, open, messages]);

  useEffect(() => {
    if (open && inputRef.current) {
      const timer = setTimeout(() => inputRef.current.focus(), 60);
      return () => clearTimeout(timer);
    }
    return undefined;
  }, [open]);

  // Resolve the property the user is looking at, so the assistant answers about
  // that listing with the same numbers the page shows.
  useEffect(() => {
    if (!open || !propertyId) {
      setContext(null);
      return undefined;
    }
    let cancelled = false;
    Promise.all([
      propertyById(propertyId).catch(() => null),
      propertyChain(propertyId).catch(() => null),
    ]).then(([detail, chain]) => {
      if (cancelled) return;
      setContext(buildContext(detail, chain));
    });
    return () => {
      cancelled = true;
    };
  }, [open, propertyId]);

  const setOpen = (next) => {
    if (onOpenChange) onOpenChange(next);
  };

  const toggle = () => setOpen(!open);

  const send = async (override) => {
    const text = (override ?? draft).trim();
    if (!text || busy) return;
    if (!override) setDraft("");
    setBusy(true);

    const userMessage = { role: "user", content: text };
    const nextMessages = [...messages, userMessage];
    // While the reply is in flight the typing bubble should be the thing that
    // stays in view; only once it lands do we anchor on its first line.
    revealTopRef.current = false;
    setMessages(nextMessages);

    try {
      const response = await fetch(`${AI_BASE}${ENDPOINT}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          messages: nextMessages,
          context: context ? context.text : null,
        }),
      });
      const data = await response.json();
      const reply = data.reply?.trim() || "I didn't get that. Could you rephrase?";
      revealTopRef.current = true;
      setMessages([...nextMessages, { role: "assistant", content: reply }]);
      setProvider(data.provider || (data.offline ? "offline" : null));
      // The local model answers correctly but takes a couple of minutes, so a
      // degraded reply is flagged rather than left looking like a normal answer.
      setDegraded(!!data.degraded);
    } catch {
      setMessages([
        ...nextMessages,
        {
          role: "assistant",
          content:
            "I couldn't reach the MILLOW AI server. Make sure it's running with `npm run ai`.",
        },
      ]);
      revealTopRef.current = true;
      setProvider("offline");
      setDegraded(true);
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
    setDegraded(false);
  };

  const actions = actionsFor(context);

  return (
    <div className="chatbot">
      {open && (
        <div className="chatbot__panel">
          <div className="chatbot__header">
            <div className="chatbot__title">
              <span className="chatbot__avatar">AI</span>
              <div>
                <strong>MILLOW AI</strong>
                <span className="chatbot__status">
                  {busy
                    ? "Thinking..."
                    : degraded
                      ? "Slow local model (Groq unavailable)"
                      : provider
                        ? `via ${provider}`
                        : context
                          ? `asking about ${context.label}`
                          : "Marketplace assistant"}
                </span>
              </div>
            </div>
            <button
              type="button"
              className="chatbot__reset"
              title="Clear chat"
              onClick={reset}
            >
              Clear
            </button>
          </div>

          <div className="chatbot__body" ref={bodyRef}>
            {messages.map((message, index) => (
              <div
                key={index}
                ref={
                  index === messages.length - 1 ? lastMsgRef : undefined
                }
                className={`chatbot__msg chatbot__msg--${message.role}`}
              >
                {renderMessage(message.content)}
              </div>
            ))}
            {busy && (
              <div
                className="chatbot__typing"
                role="status"
                aria-label="MILLOW AI is typing"
              >
                <span className="chatbot__dot" />
                <span className="chatbot__dot" />
                <span className="chatbot__dot" />
              </div>
            )}
          </div>

          <div className="chatbot__chips">
            {actions.map((action) => (
              <button
                key={action.id}
                type="button"
                className="chatbot__chip"
                disabled={busy}
                onClick={() => send(action.prompt)}
              >
                {action.icon}
                {action.label}
              </button>
            ))}
          </div>

          {context && (
            <p className="chatbot__context">
              Asking about <strong>{context.label}</strong> ·{" "}
              {context.mreid_id}
            </p>
          )}

          <div className="chatbot__footer">
            <textarea
              ref={inputRef}
              className="chatbot__input"
              rows="1"
              placeholder="Ask about a property, price, market or transaction..."
              value={draft}
              onChange={(e) => setDraft(e.target.value)}
              onKeyDown={onKeyDown}
              disabled={busy}
            />
            <button
              type="button"
              className="chatbot__send"
              onClick={() => send()}
              disabled={busy || !draft.trim()}
            >
              Send
            </button>
          </div>
        </div>
      )}

      <button
        type="button"
        className="chatbot__fab"
        onClick={toggle}
        title="Ask MILLOW AI about this property"
        aria-label={open ? "Close MILLOW AI assistant" : "Open MILLOW AI assistant"}
      >
        {open ? (
          <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.4" strokeLinecap="round">
            <path d="M18 6 6 18M6 6l12 12" />
          </svg>
        ) : (
          <>
            <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <path d="M3 9.75 12 3l9 6.75V20a1.5 1.5 0 0 1-1.5 1.5h-4.75V15h-5.5v6.5H4.5A1.5 1.5 0 0 1 3 20z" />
            </svg>
          </>
        )}
        {!open && unread && <span className="chatbot__badge" />}
      </button>
    </div>
  );
};

export default ChatBot;
