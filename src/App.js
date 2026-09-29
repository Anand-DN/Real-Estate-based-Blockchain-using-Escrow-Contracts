import { useEffect, useState, useCallback, useRef } from "react";
import { ethers } from "ethers";

// Components
import Navigation from "./components/Navigation";
import Dashboard from "./components/Dashboard";
import ChatBot from "./components/ChatBot";
import Marketplace from "./components/Marketplace";
import PropertyDetail from "./components/PropertyDetail";

// ABIs
import PropertyNFT from "./abis/PropertyNFT.json";
import PropertyRegistry from "./abis/PropertyRegistry.json";
import MillowEscrow from "./abis/MillowEscrow.json";

// Config
import config from "./config.json";

// Routing
import {
  MARKETPLACE,
  DASHBOARD,
  BUY,
  navigate,
  pathFor,
  useRoute,
} from "./lib/routes";

// The Marketplace is the MILLOW home page.  The only other destination is the
// user's own dashboard, plus the property details overlay and the AI assistant
// which are layered on top of the marketplace.
const HOMES_CAP = 30;

const FAVORITES_KEY = (account) => `millow_favs_${account}`;

// Favorites are MREID identifiers, matching every other marketplace surface.
// Older builds stored numeric demo token ids, which are not MREIDs and cannot
// be resolved, so they are dropped instead of being shown as broken rows.
const readFavorites = (account) => {
  if (!account) return [];
  try {
    const stored = JSON.parse(localStorage.getItem(FAVORITES_KEY(account)));
    if (!Array.isArray(stored)) return [];
    return stored.filter((id) => /^MREID_\d+$/i.test(String(id)));
  } catch {
    return [];
  }
};

function App() {
  const route = useRoute();
  const view = route.view;
  const marketSection = route.section;
  const propertyId = route.propertyId || null;

  const [provider, setProvider] = useState(null);
  const [escrow, setEscrow] = useState(null);
  const [realEstate, setRealEstate] = useState(null);
  const [registry, setRegistry] = useState(null);

  const [account, setAccount] = useState(null);

  const [homes, setHomes] = useState([]);
  const [networkError, setNetworkError] = useState(null);
  const [notification, setNotification] = useState(null);
  const [favorites, setFavorites] = useState([]);

  // The MILLOW AI launcher is available on every view, not only behind a
  // property.  With a property open the route stays the source of truth, because
  // /assistant/:mreid is a deep link that has to survive a reload; without one
  // there is no route to hang the panel off, so it is plain local state.
  const [standaloneAssistant, setStandaloneAssistant] = useState(false);
  const [theme, setTheme] = useState(() => {
    const stored = localStorage.getItem("millow_theme");
    if (stored === "dark" || stored === "light") return stored;
    return window.matchMedia &&
      window.matchMedia("(prefers-color-scheme: dark)").matches
      ? "dark"
      : "light";
  });
  const networkSwitchAttempted = useRef(false);
  const [themeCurtain, setThemeCurtain] = useState(false);

  // Where the property overlay was opened from, so closing it returns to the
  // page the user was actually on rather than always dropping to the home page.
  const propertyOrigin = useRef(MARKETPLACE);

  useEffect(() => {
    document.documentElement.setAttribute("data-theme", theme);
    localStorage.setItem("millow_theme", theme);
  }, [theme]);

  const toggleTheme = () => {
    if (themeCurtain) return;
    setThemeCurtain(true);
    setTimeout(() => setTheme((prev) => (prev === "dark" ? "light" : "dark")), 580);
    setTimeout(() => setThemeCurtain(false), 1220);
  };

  const notify = useCallback((text, type) => setNotification({ text, type }), []);

  useEffect(() => {
    if (!notification) return;
    const timer = setTimeout(() => setNotification(null), 4000);
    return () => clearTimeout(timer);
  }, [notification]);

  const goMarketplace = useCallback((section = BUY) => {
    navigate(pathFor({ view: MARKETPLACE, section }));
  }, []);

  const goDashboard = useCallback(() => {
    navigate(pathFor({ view: DASHBOARD }));
  }, []);

  const openProperty = useCallback(
    (mreidId, from = view) => {
      if (!mreidId) return;
      propertyOrigin.current = from === DASHBOARD ? DASHBOARD : MARKETPLACE;
      navigate(`/property/${encodeURIComponent(mreidId)}`);
    },
    [view],
  );

  const closeProperty = useCallback(() => {
    navigate(pathFor({ view: propertyOrigin.current, section: marketSection }));
  }, [marketSection]);

  const setAssistantOpen = useCallback(
    (open) => {
      if (!propertyId) {
        setStandaloneAssistant(open);
        return;
      }
      if (open) {
        navigate(
          pathFor({
            view,
            section: marketSection,
            propertyId,
            assistant: true,
          }),
        );
      } else {
        navigate(pathFor({ view, section: marketSection, propertyId }), {
          replace: true,
        });
      }
    },
    [propertyId, view, marketSection],
  );

  const switchNetwork = useCallback(async () => {
    if (networkSwitchAttempted.current) return;
    networkSwitchAttempted.current = true;
    try {
      await window.ethereum.request({
        method: "wallet_switchEthereumChain",
        params: [{ chainId: "0x7A69" }],
      });
    } catch (error) {
      if (error.code === 4902) {
        await window.ethereum.request({
          method: "wallet_addEthereumChain",
          params: [
            {
              chainId: "0x7A69",
              chainName: "Millow Localhost",
              rpcUrls: ["http://localhost:8545"],
            },
          ],
        });
      }
    }
  }, []);

  const loadBlockchainData = useCallback(async () => {
    if (!window.ethereum) {
      setNetworkError(
        "No wallet detected. Install a browser wallet to connect. Browsing the marketplace works without one.",
      );
      return;
    }

    try {
      const provider = new ethers.providers.Web3Provider(window.ethereum);
      setProvider(provider);

      const accounts = await window.ethereum.request({
        method: "eth_accounts",
      });
      if (accounts.length) {
        setAccount(ethers.utils.getAddress(accounts[0]));
      }

      const network = await provider.getNetwork();

      if (!config[network.chainId]) {
        setNetworkError(
          `Unsupported network (chainId ${network.chainId}). This app requires the Millow Localhost network (chainId 31337).`,
        );
        switchNetwork();
        return;
      }

      setNetworkError(null);

      const realEstate = new ethers.Contract(
        config[network.chainId].propertyNft.address,
        PropertyNFT,
        provider,
      );
      const escrow = new ethers.Contract(
        config[network.chainId].millowEscrow.address,
        MillowEscrow,
        provider,
      );
      const registry = new ethers.Contract(
        config[network.chainId].propertyRegistry.address,
        PropertyRegistry,
        provider,
      );
      setRealEstate(realEstate);
      setEscrow(escrow);
      setRegistry(registry);

      let totalSupply = 0;
      try {
        totalSupply = (await realEstate.totalSupply()).toNumber();
      } catch (err) {
        console.error("Could not fetch totalSupply from contract", err);
        throw err;
      }

      const homes = [];

      const fetchWithTimeout = async (url, ms = 4000) => {
        const controller = new AbortController();
        const timer = setTimeout(() => controller.abort(), ms);
        try {
          const response = await fetch(url, { signal: controller.signal });
          return await response.json();
        } finally {
          clearTimeout(timer);
        }
      };

      // The dashboard resolves the token metadata for the properties that have
      // on-chain activity, so only those need to be pre-scraped.  Scanning the
      // whole 29k token range on every marketplace page load would be unusable.
      const scanLimit = Math.min(totalSupply, HOMES_CAP);

      for (let i = 1; i <= scanLimit; i++) {
        try {
          const uri = await realEstate.tokenURI(i);
          const metadata = await fetchWithTimeout(uri);
          homes.push({ ...metadata, tokenId: i });
        } catch (error) {
          console.warn(`Could not load metadata for token ${i}`, error);
        }
      }

      setHomes(homes);
      // The dashboard reads the same cache; make it re-run now that the
      // metadata is present instead of waiting for the next account change.
      window.dispatchEvent(new Event("millow:dashboard-refresh"));
    } catch (error) {
      console.error(error);
      setNetworkError(
        "Could not connect to the local blockchain. Start the persistent chain with `npm run chain:start`.",
      );
    }
  }, [switchNetwork]);

  useEffect(() => {
    loadBlockchainData();
  }, [loadBlockchainData]);

  useEffect(() => {
    if (!window.ethereum) return;

    const handleAccountsChanged = (accounts) => {
      if (accounts.length === 0) {
        setAccount(null);
      } else {
        setAccount(ethers.utils.getAddress(accounts[0]));
      }
    };

    const handleChainChanged = () => {
      loadBlockchainData();
    };

    window.ethereum.on("accountsChanged", handleAccountsChanged);
    window.ethereum.on("chainChanged", handleChainChanged);

    return () => {
      window.ethereum.removeListener("accountsChanged", handleAccountsChanged);
      window.ethereum.removeListener("chainChanged", handleChainChanged);
    };
  }, [loadBlockchainData]);

  // Do not reload the entire app from escrow events.  Replacing the contract
  // instance from inside an event callback also replaces this listener, which
  // can create a refresh loop while the dashboard is open.  Transaction views
  // update their own state, and account/network changes still reload below.

  useEffect(() => {
    setFavorites(readFavorites(account));
  }, [account]);

  const toggleFavorite = (mreidId) => {
    setFavorites((prev) => {
      const next = prev.includes(mreidId)
        ? prev.filter((id) => id !== mreidId)
        : [...prev, mreidId];
      if (account) {
        localStorage.setItem(FAVORITES_KEY(account), JSON.stringify(next));
      }
      return next;
    });
  };

  // Pressing Escape closes every overlay.  The property overlay and the
  // assistant both live on top of the marketplace, so the marketplace stays the
  // page Escape returns to.
  useEffect(() => {
    const onEscape = (e) => {
      if (e.key !== "Escape") return;
      setAssistantOpen(false);
    };
    window.addEventListener("keydown", onEscape);
    return () => window.removeEventListener("keydown", onEscape);
  }, [setAssistantOpen]);

  return (
    <div>
      <Navigation
        account={account}
        setAccount={setAccount}
        view={view}
        section={marketSection}
        onGoMarketplace={goMarketplace}
        onGoDashboard={goDashboard}
        onGoHome={goMarketplace}
        theme={theme}
        onToggleTheme={toggleTheme}
      />

      {networkError && (
        <div className="network__error" role="status">
          {networkError}
        </div>
      )}

      {notification && (
        <div
          className={`toast ${
            notification.type === "success" ? "toast--success" : "toast--error"
          }`}
        >
          {notification.text}
          <button
            type="button"
            className="toast__close"
            onClick={() => setNotification(null)}
          >
            x
          </button>
        </div>
      )}

      {view === MARKETPLACE ? (
        <Marketplace
          section={marketSection}
          setSection={goMarketplace}
          favorites={favorites}
          onToggleFavorite={toggleFavorite}
          account={account}
          provider={provider}
          realEstate={realEstate}
          escrow={escrow}
          onSelectProperty={openProperty}
        />
      ) : (
        <Dashboard
          account={account}
          realEstate={realEstate}
          escrow={escrow}
          registry={registry}
          homes={homes}
          favorites={favorites}
          onToggleFavorite={toggleFavorite}
          onSelectProperty={openProperty}
          setNotification={notify}
        />
      )}

      {propertyId && (
        <PropertyDetail
          key={propertyId}
          mreidId={propertyId}
          onClose={closeProperty}
          onSelectSimilar={openProperty}
        />
      )}

      <ChatBot
        open={propertyId ? Boolean(route.assistant) : standaloneAssistant}
        onOpenChange={setAssistantOpen}
        propertyId={propertyId}
      />

      {themeCurtain && (
        <div className="theme-curtain" aria-hidden="true">
          <span className="theme-curtain__panel theme-curtain__panel--left" />
          <span className="theme-curtain__panel theme-curtain__panel--right" />
        </div>
      )}
    </div>
  );
}

export default App;
