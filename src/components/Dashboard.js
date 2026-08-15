import { useEffect, useState, useRef } from "react";
import { ethers } from "ethers";

const RPC_TIMEOUT = 5000;

const Dashboard = ({
  account,
  realEstate,
  escrow,
  homes,
  onSelect,
  setNotification,
}) => {
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [role, setRole] = useState(null);

  const [owned, setOwned] = useState([]);
  const [buying, setBuying] = useState([]);
  const [selling, setSelling] = useState([]);
  const [sold, setSold] = useState([]);
  const [purchases, setPurchases] = useState([]);
  const [inspected, setInspected] = useState([]);
  const [lended, setLended] = useState([]);

  const [aiMarket, setAiMarket] = useState(null);
  const [aiLoading, setAiLoading] = useState(false);
  const [aiError, setAiError] = useState(false);
  const [recs, setRecs] = useState([]);

  const AI_BASE = process.env.REACT_APP_AI_URL || "http://localhost:8000";

  const findHome = (tokenId) =>
    homes.find(
      (h) => Number(h.tokenId) === tokenId || Number(h.id) === tokenId,
    );

  const loadRef = useRef(null);
  const homesRef = useRef(homes);
  const requestRef = useRef({ key: null, running: false, completed: false });

  useEffect(() => {
    homesRef.current = homes;
  }, [homes]);

  useEffect(() => {
    let cancelled = false;
    setAiLoading(true);
    setAiError(false);
    fetch(`${AI_BASE}/market/insights`)
      .then((res) => (res.ok ? res.json() : Promise.reject(new Error())))
      .then((data) => {
        if (!cancelled) {
          setAiMarket(data);
          setAiLoading(false);
        }
      })
      .catch(() => {
        if (!cancelled) {
          setAiError(true);
          setAiLoading(false);
        }
      });
    return () => {
      cancelled = true;
    };
  }, [AI_BASE]);

  useEffect(() => {
    if (loading) return;
    const sources = [...owned, ...buying];
    if (!sources.length) {
      setRecs([]);
      return;
    }
    let cancelled = false;
    const seen = {};
    const all = [];
    Promise.all(
      sources.map(async (p) => {
        const tokenId = Number(p.meta?.tokenId ?? p.meta?.id);
        if (!tokenId) return;
        try {
          const res = await fetch(`${AI_BASE}/recommendations/${tokenId}`);
          if (!res.ok) return;
          const data = await res.json();
          for (const similar of data.similar) {
            if (seen[similar.token_id]) {
              seen[similar.token_id].match = Math.max(
                seen[similar.token_id].match,
                similar.match,
              );
            } else {
              seen[similar.token_id] = similar;
              all.push(similar);
            }
          }
        } catch {}
      }),
    ).then(() => {
      if (!cancelled) {
        setRecs(all.sort((a, b) => b.match - a.match).slice(0, 3));
      }
    });
    return () => {
      cancelled = true;
    };
  }, [owned, buying, loading, AI_BASE]);

  useEffect(() => {
    const refresh = () => {
      requestRef.current.key = null;
      requestRef.current.completed = false;
      if (loadRef.current) loadRef.current();
    };
    window.addEventListener("millow:dashboard-refresh", refresh);
    return () =>
      window.removeEventListener("millow:dashboard-refresh", refresh);
  }, []);

  // ---------------------------------------------------------
  // Utility: prevent a blockchain call from hanging forever
  // ---------------------------------------------------------

  const withTimeout = (promise, ms = RPC_TIMEOUT) => {
    return Promise.race([
      promise,
      new Promise((_, reject) => {
        setTimeout(() => {
          reject(new Error("Blockchain request timed out"));
        }, ms);
      }),
    ]);
  };

  // ---------------------------------------------------------
  // LOAD DASHBOARD
  // ---------------------------------------------------------

  useEffect(() => {
    let mounted = true;

    const loadDashboard = async () => {
      if (!account || !realEstate || !escrow) {
        if (mounted) {
          setLoading(false);
        }
        return;
      }

      // Contract instances may be recreated when the application refreshes its
      // connection.  Only load once for the same account/contracts pair; without
      // this guard every refresh starts another dashboard load and it never gets
      // a chance to leave the loading state.
      const requestKey = `${account.toLowerCase()}:${realEstate.address}:${
        escrow.address
      }`;
      if (requestRef.current.key === requestKey) {
        if (requestRef.current.completed && mounted) setLoading(false);
        return;
      }
      requestRef.current = { key: requestKey, running: true, completed: false };

      setLoading(true);
      setError(null);
      setRole(null);

      try {
        console.log("================================");
        console.log("Loading Millow Dashboard");
        console.log("Account:", account);
        console.log("================================");

        // -------------------------------------------------
        // Get total supply
        // -------------------------------------------------

        const totalSupplyBN = await withTimeout(realEstate.totalSupply());

        const totalSupply = totalSupplyBN.toNumber();

        console.log("Total properties:", totalSupply);

        // -------------------------------------------------
        // Get global Inspector and Lender addresses
        // -------------------------------------------------

        const inspectorAddr = await withTimeout(escrow.inspector());

        const lenderAddr = await withTimeout(escrow.lender());

        const acct = account.toLowerCase();

        console.log("Inspector:", inspectorAddr);
        console.log("Lender:", lenderAddr);

        // -------------------------------------------------
        // Arrays
        // -------------------------------------------------

        const ownedList = [];
        const buyingList = [];
        const sellingList = [];
        const soldList = [];
        const inspectedList = [];
        const lendedList = [];

        let isSeller = false;
        let isBuyer = false;

        // -------------------------------------------------
        // Load properties
        //
        // IMPORTANT:
        // We do this one property at a time.
        // This avoids sending dozens of RPC calls
        // simultaneously to Hardhat.
        // -------------------------------------------------

        for (let id = 1; id <= totalSupply; id++) {
          if (!mounted) return;

          try {
            console.log(`Loading property ${id}...`);

            const owner = await withTimeout(realEstate.ownerOf(id));

            const isListed = await withTimeout(escrow.isListed(id));

            const seller = await withTimeout(escrow.sellerOf(id));

            const buyer = await withTimeout(escrow.buyer(id));

            const inspectionPassed = await withTimeout(
              escrow.inspectionPassed(id),
            );

            const lenderApproved = await withTimeout(
              escrow.approval(id, lenderAddr),
            );

            const price = await withTimeout(escrow.purchasePrice(id));

            const escrowAmt = await withTimeout(escrow.escrowAmount(id));

            // -------------------------------------------------
            // Find metadata using tokenId first
            // -------------------------------------------------

            const meta =
              homesRef.current.find(
                (h) => Number(h.tokenId) === id || Number(h.id) === id,
              ) || homesRef.current[id - 1];

            if (!meta) {
              console.warn(`No metadata found for property ${id}`);
              continue;
            }

            const property = {
              id,
              owner,
              isListed,
              seller,
              buyer,
              inspectionPassed,
              lenderApproved,
              price,
              escrowAmt,
              meta,
            };

            const sellerMatch =
              seller &&
              seller !== ethers.constants.AddressZero &&
              seller.toLowerCase() === acct;

            const buyerMatch =
              buyer &&
              buyer !== ethers.constants.AddressZero &&
              buyer.toLowerCase() === acct;

            // -------------------------------------------------
            // SELLER
            // -------------------------------------------------

            if (sellerMatch) {
              isSeller = true;

              // Seller's currently listed properties
              if (isListed) {
                sellingList.push(property);
              }

              // Seller's completed sales
              if (!isListed && owner.toLowerCase() !== acct) {
                soldList.push(property);
              }
            }

            // -------------------------------------------------
            // BUYER
            // -------------------------------------------------

            if (buyerMatch) {
              isBuyer = true;

              // Currently buying / in escrow
              if (isListed) {
                buyingList.push(property);
              }

              // Completed purchase
              if (!isListed && owner.toLowerCase() === acct) {
                ownedList.push(property);
              }
            }

            // -------------------------------------------------
            // INSPECTOR
            // -------------------------------------------------

            if (inspectionPassed) {
              inspectedList.push(property);
            }

            // -------------------------------------------------
            // LENDER
            // -------------------------------------------------

            if (lenderApproved) {
              lendedList.push(property);
            }
          } catch (propertyError) {
            console.warn(`Could not load property ${id}:`, propertyError);

            // Continue loading the other properties
          }
        }

        if (!mounted) return;

        // -------------------------------------------------
        // Set dashboard data
        // -------------------------------------------------

        setOwned(ownedList);
        setBuying(buyingList);
        setSelling(sellingList);
        setSold(soldList);
        setInspected(inspectedList);
        setLended(lendedList);

        /*
         * IMPORTANT:
         *
         * We are NOT using queryFilter(Transfer) here.
         *
         * That query was causing unnecessary RPC delays.
         *
         * Purchase history will be added later using a
         * better method.
         */
        setPurchases([]);

        // -------------------------------------------------
        // Determine role
        // -------------------------------------------------

        if (acct === inspectorAddr.toLowerCase()) {
          setRole("Inspector");
        } else if (acct === lenderAddr.toLowerCase()) {
          setRole("Lender");
        } else if (isSeller) {
          setRole("Seller");
        } else if (isBuyer) {
          setRole("Buyer");
        } else {
          /*
           * New/unassigned account.
           *
           * We don't call it Seller just because it
           * owns an NFT.
           */
          setRole("Buyer");
        }

        console.log("Seller:", isSeller);
        console.log("Buyer:", isBuyer);
        console.log(
          "Final role:",
          isSeller ? "Seller" : isBuyer ? "Buyer" : "Buyer",
        );

        console.log("Dashboard loaded successfully.");

        setError(null);
        requestRef.current = {
          key: requestKey,
          running: false,
          completed: true,
        };
      } catch (err) {
        console.error("Dashboard loading error:", err);

        if (mounted) {
          setError(
            "Could not load your dashboard. Make sure Hardhat is running on http://localhost:8545 and MetaMask is connected to chain 31337.",
          );

          if (setNotification) {
            setNotification("Failed to load dashboard", "error");
          }
        }
      } finally {
        if (requestRef.current.key === requestKey) {
          requestRef.current.running = false;
        }
        if (mounted) {
          setLoading(false);
        }
      }
    };

    loadRef.current = loadDashboard;

    loadDashboard();

    // Keep an in-flight request alive across React's development effect replay.
    // Cancelling it here leaves the replacement effect waiting on a request that
    // can no longer update state.
    return undefined;

    // IMPORTANT:
    // Do not add setNotification here.
    // App recreates notify() on every render.
  }, [account, realEstate, escrow]);

  // ---------------------------------------------------------
  // SECTION COMPONENT
  // ---------------------------------------------------------

  const Section = ({ title, badge, items, empty, amount = false }) => {
    return (
      <div className="dash__section">
        <div className="dash__section-head">
          <h3>{title}</h3>

          {items.length > 0 && (
            <span className={`dash__badge dash__badge--${badge}`}>
              {items.length}
            </span>
          )}
        </div>

        {items.length === 0 ? (
          <p className="cards__empty">{empty}</p>
        ) : (
          <div className="cards">
            {items.map((item, index) => {
              const attributes = item.meta?.attributes || [];

              const purchasePrice = attributes[0]?.value ?? "-";

              const bedrooms = attributes[2]?.value ?? "-";

              const bathrooms = attributes[3]?.value ?? "-";

              const sqft = attributes[4]?.value ?? "-";

              return (
                <div
                  className="card"
                  key={item.id}
                  style={{
                    animationDelay: `${index * 0.05}s`,
                  }}
                  onClick={() => onSelect && onSelect(item.meta)}
                >
                  <div className="card__image">
                    <img
                      src={item.meta.image || "/images/1.jpg"}
                      alt={item.meta.name || "Home"}
                    />
                  </div>

                  <div className="card__info">
                    <h4>{purchasePrice} ETH</h4>

                    {amount && (
                      <p className="dash__amount">
                        <strong>Lent:</strong>{" "}
                        {(() => {
                          try {
                            return Number(
                              ethers.utils.formatEther(
                                item.price.sub(item.escrowAmt),
                              ),
                            ).toFixed(2);
                          } catch {
                            return "0.00";
                          }
                        })()}{" "}
                        ETH
                      </p>
                    )}

                    <p>
                      <strong>{bedrooms}</strong> bds |{" "}
                      <strong>{bathrooms}</strong> ba | <strong>{sqft}</strong>{" "}
                      sqft
                    </p>

                    <p>{item.meta.address || "Address unavailable"}</p>
                  </div>

                  <span className={`dash__tag dash__tag--${badge}`}>
                    {badge}
                  </span>
                </div>
              );
            })}
          </div>
        )}
      </div>
    );
  };

  // ---------------------------------------------------------
  // SHORT ACCOUNT
  // ---------------------------------------------------------

  const shortAccount = account
    ? `${account.slice(0, 6)}...${account.slice(-4)}`
    : null;

  // ---------------------------------------------------------
  // NO ACCOUNT
  // ---------------------------------------------------------

  if (!account) {
    return (
      <div className="dash">
        <div className="dash__hero">
          <h2>My Dashboard</h2>

          <p>Connect your wallet to view your dashboard.</p>
        </div>
      </div>
    );
  }

  // ---------------------------------------------------------
  // CONTRACTS NOT READY
  // ---------------------------------------------------------

  if (!realEstate || !escrow) {
    return (
      <div className="dash">
        <div className="dash__hero">
          <h2>My Dashboard</h2>

          <p>{shortAccount}</p>
        </div>

        <div className="dash__error">
          <p>
            Waiting for blockchain connection. Make sure Hardhat is running on
            http://localhost:8545 and MetaMask is connected to chain 31337.
          </p>
        </div>
      </div>
    );
  }

  // ---------------------------------------------------------
  // MAIN DASHBOARD
  // ---------------------------------------------------------

  return (
    <div className="dash">
      <div className="dash__hero">
        <h2>My Dashboard</h2>

        <p>{shortAccount}</p>

        {role && (
          <span className={`dash__role dash__role--${role.toLowerCase()}`}>
            {role}
          </span>
        )}
      </div>

      {/* AI MARKET INSIGHTS */}

      <div className="dash__section dash__ai">
        <div className="dash__section-head">
          <h3>AI Market Insights</h3>
        </div>

        {aiLoading ? (
          <p className="cards__empty">Loading AI market analysis...</p>
        ) : aiError ? (
          <p className="cards__empty">
            AI insights unavailable. Start the AI server with{" "}
            <code>npm run ai</code> and refresh.
          </p>
        ) : aiMarket ? (
          <>
            <div className="ai__stats">
              <div className="ai__stat">
                <span>Avg listed</span>
                <strong>{aiMarket.stats.avg_listed_eth} ETH</strong>
              </div>
              <div className="ai__stat">
                <span>Avg AI value</span>
                <strong>{aiMarket.stats.avg_predicted_eth} ETH</strong>
              </div>
              <div className="ai__stat">
                <span>Undervalued</span>
                <strong>{aiMarket.stats.undervalued}</strong>
              </div>
              <div className="ai__stat">
                <span>Overvalued</span>
                <strong>{aiMarket.stats.overvalued}</strong>
              </div>
              <div className="ai__stat">
                <span>Avg risk</span>
                <strong>{aiMarket.stats.avg_risk}/100</strong>
              </div>
            </div>

            <h4 className="ai__subtitle">Top undervalued picks</h4>
            <div className="cards">
              {aiMarket.deals.map((deal, index) => (
                <div
                  className="card"
                  key={deal.token_id}
                  style={{ animationDelay: `${index * 0.05}s` }}
                  onClick={() => {
                    const home = findHome(deal.token_id);
                    if (home) onSelect(home);
                  }}
                >
                  <div className="card__image">
                    <img src={deal.image} alt={deal.name} />
                  </div>
                  <div className="card__info">
                    <h4>{deal.listed_price_eth} ETH</h4>
                    <p className="ai__match">
                      {deal.verdict} · {deal.difference_pct >= 0 ? "+" : ""}
                      {deal.difference_pct}%
                    </p>
                    <p>{deal.name}</p>
                  </div>
                  <span className="dash__tag dash__tag--selling">
                    {deal.verdict}
                  </span>
                </div>
              ))}
            </div>
          </>
        ) : null}
      </div>

      {/* RECOMMENDED FOR YOU */}

      {recs.length > 0 && (
        <div className="dash__section dash__ai">
          <div className="dash__section-head">
            <h3>Recommended for you</h3>
          </div>
          <div className="cards">
            {recs.map((similar, index) => (
              <div
                className="card"
                key={similar.token_id}
                style={{ animationDelay: `${index * 0.05}s` }}
                onClick={() => {
                  const home = findHome(similar.token_id);
                  if (home) onSelect(home);
                }}
              >
                <div className="card__image">
                  <img src={similar.image} alt={similar.name} />
                </div>
                <div className="card__info">
                  <h4>{similar.price_eth} ETH</h4>
                  <p className="ai__match">{similar.match}% match</p>
                  <p>{similar.name}</p>
                </div>
                <span className="dash__tag dash__tag--buying">AI pick</span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* ERROR */}

      {error ? (
        <div className="dash__error">
          <p>{error}</p>

          <button
            type="button"
            className="dash__retry"
            onClick={() => {
              requestRef.current.completed = false;
              requestRef.current.key = null;
              if (loadRef.current) {
                loadRef.current();
              }
            }}
          >
            Retry
          </button>
        </div>
      ) : loading ? (
        /* LOADING */

        <div className="dash__loading">
          <div className="spinner"></div>

          <p>Loading your dashboard...</p>
        </div>
      ) : (
        /* LOADED */

        <>
          {/* ============================= */}
          {/* INSPECTOR */}
          {/* ============================= */}

          {role === "Inspector" && (
            <Section
              title="Properties inspected"
              badge="inspected"
              items={inspected}
              empty="No inspections completed yet."
            />
          )}

          {/* ============================= */}
          {/* LENDER */}
          {/* ============================= */}

          {role === "Lender" && (
            <Section
              title="Loans funded"
              badge="lended"
              items={lended}
              empty="No loans funded yet."
              amount={true}
            />
          )}

          {/* ============================= */}
          {/* SELLER */}
          {/* ============================= */}

          {role === "Seller" && (
            <>
              <Section
                title="Properties for sale"
                badge="selling"
                items={selling}
                empty="You haven't listed any properties."
              />

              <Section
                title="Sold"
                badge="sold"
                items={sold}
                empty="No properties sold yet."
              />
            </>
          )}

          {/* ============================= */}
          {/* BUYER */}
          {/* ============================= */}

          {role === "Buyer" && (
            <>
              <Section
                title="Owned"
                badge="owned"
                items={owned}
                empty="You don't own any properties yet."
              />

              <Section
                title="Buying (in escrow)"
                badge="buying"
                items={buying}
                empty="No properties being purchased right now."
              />

              {/* <Section
                title="Purchase history"
                badge="history"
                items={purchases}
                empty="No purchase history yet."
              /> */}
            </>
          )}
        </>
      )}
    </div>
  );
};

export default Dashboard;
