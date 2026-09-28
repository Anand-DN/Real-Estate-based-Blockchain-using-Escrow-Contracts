// Minimal URL router for MILLOW.
//
// The application already models navigation as a small `view` state machine
// (App.js) and has no router dependency, so this module does not introduce one.
// It only keeps that state and the browser URL in sync, which is what makes the
// Marketplace the real landing page:
//
//   /                -> Marketplace (Buy)
//   /marketplace     -> Marketplace (Buy)
//   /buy             -> Marketplace (Buy)
//   /rent            -> Marketplace (Rent)
//   /sell            -> Marketplace (Sell)
//   /property/:mreid -> Marketplace with the property details open
//   /dashboard       -> My Dashboard
//   /assistant       -> Marketplace with the MILLOW AI assistant open
//   /assistant/:mreid-> the same property, with the assistant open
//
// Anything unknown resolves to the Marketplace, so a bad or stale URL can never
// strand the user on a blank page.

import { useEffect, useState } from "react";

export const MARKETPLACE = "marketplace";
export const DASHBOARD = "dashboard";
export const BUY = "buy";
export const RENT = "rent";
export const SELL = "sell";

const SECTIONS = [BUY, RENT, SELL];
const NAVIGATE_EVENT = "millow:navigate";

const sectionForPath = (segment) =>
  SECTIONS.includes(segment) ? segment : BUY;

export const pathFor = ({ view, section, propertyId, assistant } = {}) => {
  if (view === DASHBOARD) return "/dashboard";
  if (view !== MARKETPLACE) return "/";
  if (propertyId) {
    // The assistant is opened from a property, so the property has to survive
    // in the URL or the launcher would close the page it belongs to.
    return assistant
      ? `/assistant/${encodeURIComponent(propertyId)}`
      : `/property/${encodeURIComponent(propertyId)}`;
  }
  if (assistant) return "/assistant";
  if (section === RENT) return "/rent";
  if (section === SELL) return "/sell";
  return "/";
};

// Turns any pathname into the application state it maps to.  Pure, so it can be
// called both on mount and on every history event.
export const parseRoute = (pathname) => {
  const clean = String(pathname || "/").split("?")[0].split("#")[0];
  const parts = clean.split("/").filter(Boolean);

  if (parts[0] === "dashboard") {
    return { view: DASHBOARD, section: BUY, propertyId: null };
  }
  if (parts[0] === "assistant") {
    return {
      view: MARKETPLACE,
      section: BUY,
      propertyId: parts[1] ? decodeURIComponent(parts[1]) : null,
      assistant: true,
    };
  }
  if (parts[0] === "property" && parts[1]) {
    return {
      view: MARKETPLACE,
      section: BUY,
      propertyId: decodeURIComponent(parts[1]),
    };
  }
  if (parts.length === 0 || parts[0] === "marketplace") {
    return { view: MARKETPLACE, section: sectionForPath(parts[1]), propertyId: null };
  }
  if (parts.length === 1) {
    return { view: MARKETPLACE, section: sectionForPath(parts[0]), propertyId: null };
  }
  return { view: MARKETPLACE, section: BUY, propertyId: null };
};

export const currentPath = () =>
  typeof window === "undefined" || !window.location
    ? "/"
    : window.location.pathname;

export const routeNow = () => parseRoute(currentPath());

// Navigates without a full page load.  `replace` keeps the back button useful
// when a route is a normalisation (unknown path -> Marketplace).
export const navigate = (path, options = {}) => {
  if (typeof window === "undefined" || !window.history) return;
  const current = currentPath();
  if (current !== path) {
    if (options.replace) {
      window.history.replaceState({}, "", path);
    } else {
      window.history.pushState({}, "", path);
    }
  }
  window.dispatchEvent(new CustomEvent(NAVIGATE_EVENT, { detail: { path } }));
};

export const subscribe = (handler) => {
  if (typeof window === "undefined") return () => {};
  window.addEventListener("popstate", handler);
  window.addEventListener(NAVIGATE_EVENT, handler);
  return () => {
    window.removeEventListener("popstate", handler);
    window.removeEventListener(NAVIGATE_EVENT, handler);
  };
};

// The route the app should be showing right now, re-rendered on history changes.
export const useRoute = () => {
  const [route, setRoute] = useState(routeNow);

  useEffect(() => {
    const sync = () => {
      const next = routeNow();
      setRoute((prev) =>
        prev.view === next.view &&
        prev.section === next.section &&
        (prev.propertyId || null) === (next.propertyId || null) &&
        Boolean(prev.assistant) === Boolean(next.assistant)
          ? prev
          : next,
      );
    };
    sync();
    return subscribe(sync);
  }, []);

  return route;
};
