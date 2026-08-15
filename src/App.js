import { useEffect, useState, useCallback, useRef } from "react";
import { ethers } from "ethers";

// Components
import Navigation from "./components/Navigation";
import Search from "./components/Search";
import Home from "./components/Home";
import List from "./components/List";
import Dashboard from "./components/Dashboard";

// ABIs
import RealEstate from "./abis/RealEstate.json";
import Escrow from "./abis/Escrow.json";

// Config
import config from "./config.json";

function App() {
  const [provider, setProvider] = useState(null);
  const [escrow, setEscrow] = useState(null);
  const [realEstate, setRealEstate] = useState(null);

  const [account, setAccount] = useState(null);

  const [homes, setHomes] = useState([]);
  const [home, setHome] = useState({});
  const [toggle, setToggle] = useState(false);
  const [networkError, setNetworkError] = useState(null);
  const [notification, setNotification] = useState(null);
  const [search, setSearch] = useState("");
  const [showList, setShowList] = useState(false);
  const [view, setView] = useState("browse");
  const [maxPrice, setMaxPrice] = useState("");
  const [sortBy, setSortBy] = useState("default");
  const [favorites, setFavorites] = useState([]);
  const [favoritesOnly, setFavoritesOnly] = useState(false);
  const [theme, setTheme] = useState(() => {
    const stored = localStorage.getItem("millow_theme");
    if (stored === "dark" || stored === "light") return stored;
    return window.matchMedia &&
      window.matchMedia("(prefers-color-scheme: dark)").matches
      ? "dark"
      : "light";
  });
  const networkSwitchAttempted = useRef(false);

  useEffect(() => {
    document.documentElement.setAttribute("data-theme", theme);
    localStorage.setItem("millow_theme", theme);
  }, [theme]);

  const toggleTheme = () => {
    setTheme((prev) => (prev === "dark" ? "light" : "dark"));
  };

  const notify = (text, type) => setNotification({ text, type });

  useEffect(() => {
    if (!notification) return;
    const timer = setTimeout(() => setNotification(null), 4000);
    return () => clearTimeout(timer);
  }, [notification]);

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
        "No wallet detected. Please install MetaMask and connect to the Millow Localhost network.",
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
        config[network.chainId].realEstate.address,
        RealEstate,
        provider,
      );
      const escrow = new ethers.Contract(
        config[network.chainId].escrow.address,
        Escrow,
        provider,
      );
      setRealEstate(realEstate);
      setEscrow(escrow);

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

      for (let i = 1; i <= totalSupply; i++) {
        try {
          const uri = await realEstate.tokenURI(i);
          const metadata = await fetchWithTimeout(uri);
          homes.push({ ...metadata, tokenId: i });
        } catch (error) {
          console.warn(`Could not load metadata for token ${i}, using fallback`, error);
          homes.push({
            id: String(i),
            tokenId: i,
            name: `Property #${i}`,
            address: "123 Localhost St",
            description: "A luxury property on Millow.",
            image: `https://ipfs.io/ipfs/QmQUozrHLAusXDxrvstCuz1T2Q1QcQySmnkFfTioLhC4jF/${i}.json`
              ? `/images/${((i - 1) % 6) + 1}.jpg`
              : "/images/1.jpg",
            attributes: [
              { trait_type: "Purchase Price", value: 15 },
              { trait_type: "Type of Residence", value: "Real Estate" },
              { trait_type: "Bed Rooms", value: 3 },
              { trait_type: "Bathrooms", value: 2 },
              { trait_type: "Square Feet", value: 2000 },
              { trait_type: "Year Built", value: 2020 },
            ],
          });
        }
      }

      setHomes(homes);
    } catch (error) {
      console.error(error);
      setNetworkError(
        "Could not connect to the local blockchain. Make sure the Hardhat node is running on http://localhost:8545.",
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
    if (!account) {
      setFavorites([]);
      return;
    }
    try {
      const stored = JSON.parse(localStorage.getItem(`millow_favs_${account}`));
      setFavorites(Array.isArray(stored) ? stored : []);
    } catch {
      setFavorites([]);
    }
  }, [account]);

  const toggleFavorite = (tokenId) => {
    setFavorites((prev) => {
      const next = prev.includes(tokenId)
        ? prev.filter((id) => id !== tokenId)
        : [...prev, tokenId];
      if (account)
        localStorage.setItem(`millow_favs_${account}`, JSON.stringify(next));
      return next;
    });
  };

  const togglePop = (home) => {
    setHome(home);
    toggle ? setToggle(false) : setToggle(true);
  };

  const openHome = (home) => {
    setHome(home);
    setToggle(true);
  };

  return (
    <div>
      <Navigation
        account={account}
        setAccount={setAccount}
        view={view}
        setView={setView}
        setShowList={setShowList}
        theme={theme}
        onToggleTheme={toggleTheme}
      />

      {networkError && <div className="network__error">{networkError}</div>}

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

      {view === "browse" ? (
        <>
          <Search search={search} setSearch={setSearch} />

          <div className="cards__section">
            <div className="cards__header">
              <h3>Homes For You</h3>
              <button
                type="button"
                className="list__button"
                onClick={() => setShowList(true)}
              >
                + List your home
              </button>
            </div>

            <div className="cards__toolbar">
              <input
                type="number"
                min="0"
                step="0.1"
                className="cards__filter"
                placeholder="Max price (ETH)"
                value={maxPrice}
                onChange={(e) => setMaxPrice(e.target.value)}
              />
              <select
                className="cards__filter"
                value={sortBy}
                onChange={(e) => setSortBy(e.target.value)}
              >
                <option value="default">Sort: Featured</option>
                <option value="low">Price: Low to High</option>
                <option value="high">Price: High to Low</option>
              </select>
              {account && (
                <button
                  type="button"
                  className={`cards__fav ${
                    favoritesOnly ? "cards__fav--active" : ""
                  }`}
                  onClick={() => setFavoritesOnly((prev) => !prev)}
                >
                  {favoritesOnly ? "Showing favorites" : "Show favorites"}
                </button>
              )}
            </div>

            <hr />

            <div className="cards">
              {homes
                .filter(
                  (home) =>
                    home.name.toLowerCase().includes(search.toLowerCase()) ||
                    home.address.toLowerCase().includes(search.toLowerCase()),
                )
                .filter(
                  (home) =>
                    !maxPrice ||
                    home.attributes[0].value <= parseFloat(maxPrice),
                )
                .filter(
                  (home) => !favoritesOnly || favorites.includes(home.tokenId),
                )
                .sort((a, b) => {
                  if (sortBy === "low")
                    return a.attributes[0].value - b.attributes[0].value;
                  if (sortBy === "high")
                    return b.attributes[0].value - a.attributes[0].value;
                  return 0;
                })
                .map((home, index) => (
                  <div
                    className="card"
                    key={index}
                    style={{ animationDelay: `${index * 0.08}s` }}
                    onClick={() => togglePop(home)}
                  >
                    <div className="card__image">
                      <img src={home.image} alt="Home" />
                      <button
                        type="button"
                        className={`card__heart ${
                          favorites.includes(home.tokenId)
                            ? "card__heart--active"
                            : ""
                        }`}
                        onClick={(e) => {
                          e.stopPropagation();
                          toggleFavorite(home.tokenId);
                        }}
                      >
                        {favorites.includes(home.tokenId) ? "♥" : "♡"}
                      </button>
                    </div>
                    <div className="card__info">
                      <h4>{home.attributes[0].value} ETH</h4>
                      <p>
                        <strong>{home.attributes[2].value}</strong> bds |
                        <strong>{home.attributes[3].value}</strong> ba |
                        <strong>{home.attributes[4].value}</strong> sqft
                      </p>
                      <p>{home.address}</p>
                    </div>
                  </div>
                ))}
            </div>

            {homes.length > 0 &&
              homes
                .filter(
                  (home) =>
                    home.name.toLowerCase().includes(search.toLowerCase()) ||
                    home.address.toLowerCase().includes(search.toLowerCase()),
                )
                .filter(
                  (home) =>
                    !maxPrice ||
                    home.attributes[0].value <= parseFloat(maxPrice),
                )
                .filter(
                  (home) => !favoritesOnly || favorites.includes(home.tokenId),
                ).length === 0 && (
                <p className="cards__empty">No properties match your search.</p>
              )}
          </div>
        </>
      ) : (
        <Dashboard
          account={account}
          realEstate={realEstate}
          escrow={escrow}
          homes={homes}
          onSelect={openHome}
          setNotification={notify}
        />
      )}

      {toggle && (
        <Home
          home={home}
          provider={provider}
          account={account}
          escrow={escrow}
          togglePop={togglePop}
          setNotification={notify}
        />
      )}

      {showList && (
        <List
          provider={provider}
          account={account}
          escrow={escrow}
          realEstate={realEstate}
          toggleList={() => setShowList(false)}
          setNotification={notify}
        />
      )}
    </div>
  );
}

export default App;
