import { useCallback, useEffect, useRef, useState } from "react";
import { ethers } from "ethers";
import MarketIntelligence from "./MarketIntelligence";
import PropertyCard from "./PropertyCard";
import { dashboardHoldings, dashboardOverview, propertiesByIds } from "../lib/millowApi";
import { STATUS_NAMES } from "../lib/blockchain";
import { formatInrCompact } from "../lib/format";

const RPC_TIMEOUT = 5000;
const HOLDINGS_PAGE = 12;

// MillowEscrow grants these AccessControl roles (keccak256 of the role name)
// instead of storing fixed inspector/lender addresses.
const INSPECTOR_ROLE = ethers.utils.id("INSPECTOR_ROLE");
const LENDER_ROLE = ethers.utils.id("LENDER_ROLE");

const ZERO = ethers.constants.AddressZero;
const short = (address) =>
  address && address !== ZERO
    ? `${address.slice(0, 6)}…${address.slice(-4)}`
    : "not set";

const isAccount = (value, acct) =>
  Boolean(value) && value !== ZERO && value.toLowerCase() === acct;

// A listing, purchase or inspection is described with what the chain actually
// says.  Anything the chain does not record is reported as unavailable rather
// than filled in from the demo dataset.
const saleStage = (sale, isListed) => {
  const status = Number(sale.status);
  if (isListed) return STATUS_NAMES[1] || "Listed";
  return STATUS_NAMES[status] || (Number.isFinite(status) ? `Stage ${status}` : null);
};

const Dashboard = ({
  account,
  realEstate,
  escrow,
  registry,
  favorites = [],
  onToggleFavorite,
  onSelectProperty,
  setNotification,
}) => {
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [role, setRole] = useState(null);
  const [flags, setFlags] = useState({ seller: false, buyer: false });

  const [owned, setOwned] = useState([]);
  const [buying, setBuying] = useState([]);
  const [selling, setSelling] = useState([]);
  const [sold, setSold] = useState([]);
  const [inspected, setInspected] = useState([]);
  const [lended, setLended] = useState([]);

  const [overview, setOverview] = useState(null);
  const [overviewError, setOverviewError] = useState(null);

  const [holdings, setHoldings] = useState(null);
  const [holdingsPage, setHoldingsPage] = useState(0);
  const [holdingsError, setHoldingsError] = useState(null);
  const [holdingsLoading, setHoldingsLoading] = useState(false);

  const [favItems, setFavItems] = useState([]);
  const [favLoading, setFavLoading] = useState(false);

  const loadRef = useRef(null);
  const requestRef = useRef({ key: null, running: false, completed: false });
  const mreidCache = useRef({});

  // ---------------------------------------------------------
  // CATALOGUE OVERVIEW (real MREID + chain snapshot numbers)
  // ---------------------------------------------------------

  const loadOverview = useCallback(() => {
    let cancelled = false;
    dashboardOverview()
      .then((data) => {
        if (cancelled) return;
        setOverview(data);
        setOverviewError(null);
      })
      .catch((err) => {
        if (cancelled) return;
        setOverview(null);
        setOverviewError(err.message);
      });
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => loadOverview(), [loadOverview]);

  // ---------------------------------------------------------
  // WALLET HOLDINGS (chain snapshot joined with the catalogue)
  // ---------------------------------------------------------

  const loadHoldings = useCallback(
    (page) => {
      if (!account) {
        setHoldings(null);
        return undefined;
      }
      let cancelled = false;
      setHoldingsLoading(true);
      dashboardHoldings(account, { limit: HOLDINGS_PAGE, offset: page * HOLDINGS_PAGE })
        .then((data) => {
          if (cancelled) return;
          setHoldings(data);
          setHoldingsError(data.available ? null : data.note);
          setHoldingsLoading(false);
        })
        .catch((err) => {
          if (cancelled) return;
          setHoldings(null);
          setHoldingsError(err.message);
          setHoldingsLoading(false);
        });
      return () => {
        cancelled = true;
      };
    },
    [account],
  );

  useEffect(() => {
    setHoldingsPage(0);
  }, [account]);

  useEffect(() => loadHoldings(holdingsPage), [loadHoldings, holdingsPage]);

  // ---------------------------------------------------------
  // FAVOURITES (stored per wallet, resolved against the registry)
  // ---------------------------------------------------------

  useEffect(() => {
    if (!favorites.length) {
      setFavItems([]);
      return undefined;
    }
    let cancelled = false;
    setFavLoading(true);
    propertiesByIds(favorites).then((rows) => {
      if (cancelled) return;
      setFavItems(rows);
      setFavLoading(false);
    });
    return () => {
      cancelled = true;
    };
  }, [favorites]);

  // ---------------------------------------------------------
  // Utility: prevent a blockchain call from hanging forever
  // ---------------------------------------------------------

  const withTimeout = (promise, ms = RPC_TIMEOUT) =>
    Promise.race([
      promise,
      new Promise((_, reject) => {
        setTimeout(
          () => reject(new Error("Blockchain request timed out")),
          ms,
        );
      }),
    ]);

  // The MREID behind a token id.  The contract is the source of truth; the
  // value is cached because the same token shows up in several sections.
  const mreidFor = useCallback(
    async (tokenId) => {
      if (mreidCache.current[tokenId]) return mreidCache.current[tokenId];
      let mreid = null;
      try {
        mreid = await withTimeout(realEstate.propertyOf(tokenId));
      } catch (err) {
        console.warn(`No MREID for token ${tokenId}:`, err);
      }
      if (mreid) mreidCache.current[tokenId] = mreid;
      return mreid;
    },
    [realEstate],
  );

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
  // LOAD DASHBOARD
  // ---------------------------------------------------------

  useEffect(() => {
    let mounted = true;

    const loadDashboard = async () => {
      if (!account || !realEstate || !escrow || !registry) {
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
      }:${registry.address}`;
      if (requestRef.current.key === requestKey) {
        if (requestRef.current.completed && mounted) setLoading(false);
        return;
      }
      requestRef.current = { key: requestKey, running: true, completed: false };

      setLoading(true);
      setError(null);
      setRole(null);

      try {
        const acct = account.toLowerCase();

        // -------------------------------------------------
        // Detect whether the connected account is the global
        // Inspector or Lender (AccessControl roles).
        // -------------------------------------------------

        const [acctIsInspector, acctIsLender] = await Promise.all([
          withTimeout(escrow.hasRole(INSPECTOR_ROLE, acct)),
          withTimeout(escrow.hasRole(LENDER_ROLE, acct)),
        ]);

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
        // Discover the tokens this dashboard cares about
        // -------------------------------------------------
        //
        // Scanning every token up to totalSupply is far too
        // slow now that the full dataset (~29k properties)
        // is tokenized.  Instead we pre-scrape the small set
        // of token ids that have any on-chain activity on the
        // PropertyRegistry or the MillowEscrow — a listing,
        // an earnest deposit, a settlement, a cancellation.
        // Only those candidates are then detail-loaded (owner,
        // escrow sale, listing) below, which keeps the number
        // of RPC calls bounded by actual activity instead of
        // by total supply.

        const candidateSet = new Set();
        const queryEvents = async (contract, topic) => {
          let filter;
          try {
            filter = contract.filters[topic]();
          } catch {
            return;
          }
          const logs = await withTimeout(contract.queryFilter(filter));
          for (const log of logs || []) {
            const tokenId = Number(log.args && log.args.tokenId);
            if (Number.isFinite(tokenId) && tokenId > 0) {
              candidateSet.add(tokenId);
            }
          }
        };

        const escrowTopics = [
          "SaleListed",
          "EarnestDeposited",
          "BalanceFunded",
          "SaleFinalized",
          "SaleCancelled",
        ];
        const registryTopics = ["Listed", "Unlisted"];

        try {
          await Promise.all([
            ...escrowTopics.map((topic) =>
              queryEvents(escrow, topic).catch((err) =>
                console.warn(`Could not load ${topic} events:`, err),
              ),
            ),
            ...registryTopics.map((topic) =>
              queryEvents(registry, topic).catch((err) =>
                console.warn(`Could not load ${topic} events:`, err),
              ),
            ),
          ]);
        } catch (err) {
          console.warn("Could not pre-scrape dashboard events:", err);
        }

        const candidates = Array.from(candidateSet).sort((a, b) => a - b);

        // -------------------------------------------------
        // Load candidate properties
        //
        // IMPORTANT:
        // We still do this one property at a time.
        // This avoids sending dozens of RPC calls
        // simultaneously to the local node.
        // -------------------------------------------------

        for (const id of candidates) {
          if (!mounted) return;

          try {
            const owner = await withTimeout(realEstate.ownerOf(id));

            const isListed = await withTimeout(escrow.isListed(id));

            const sale = await withTimeout(escrow.sales(id));

            const seller = sale.seller;
            const buyer = sale.buyer;
            const inspectionPassed = sale.inspectionPassed;
            const lenderApproved = sale.lenderFundedWei.gt(0);
            const price = sale.priceWei;
            const escrowAmt = sale.earnestWei;
            const lenderFunded = sale.lenderFundedWei;

            // The MREID is what makes a row meaningful: it is how the
            // catalogue, the AI estimate and the property page are addressed.
            const mreid = await mreidFor(id);

            const property = {
              id,
              mreid,
              owner,
              isListed,
              seller,
              buyer,
              inspectionPassed,
              lenderApproved,
              price,
              escrowAmt,
              lenderFunded,
              stage: saleStage(sale, isListed),
            };

            const sellerMatch = isAccount(seller, acct);
            const buyerMatch = isAccount(buyer, acct);

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

        setOwned(ownedList);
        setBuying(buyingList);
        setSelling(sellingList);
        setSold(soldList);
        setInspected(inspectedList);
        setLended(lendedList);
        setFlags({ seller: isSeller, buyer: isBuyer });

        // -------------------------------------------------
        // Determine role
        // -------------------------------------------------

        if (acctIsInspector) {
          setRole("Inspector");
        } else if (acctIsLender) {
          setRole("Lender");
        } else if (isSeller) {
          setRole("Seller");
        } else {
          /*
           * New/unassigned account.
           *
           * We don't call it Seller just because it
           * owns an NFT.  A wallet with no escrow
           * history is a Buyer; the sections it gets
           * are composed from `flags` below.
           */
          setRole("Buyer");
        }

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
            "Could not load your on-chain activity. Make sure the MILLOW chain is running on http://localhost:8545 and the wallet is connected to chain 31337.",
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
  }, [account, realEstate, escrow, registry, mreidFor, setNotification]);

  const retry = () => {
    requestRef.current.completed = false;
    requestRef.current.key = null;
    if (loadRef.current) loadRef.current();
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

          <p>
            Connect your wallet to see the properties you own, your listings and
            your transactions. Browsing the marketplace works without a wallet.
          </p>
        </div>
      </div>
    );
  }

  // ---------------------------------------------------------
  // CONTRACTS NOT READY
  // ---------------------------------------------------------

  if (!realEstate || !escrow || !registry) {
    return (
      <div className="dash">
        <div className="dash__hero">
          <h2>My Dashboard</h2>

          <p>{shortAccount}</p>
        </div>

        <div className="dash__error">
          <p>
            Waiting for blockchain connection. Make sure the MILLOW chain is
            running on http://localhost:8545 and the wallet is connected to
            chain 31337.
          </p>
        </div>
      </div>
    );
  }

  // ---------------------------------------------------------
  // SECTIONS
  // ---------------------------------------------------------

  const Kpi = ({ label, value, sub }) => (
    <div className="dash__kpi">
      <span className="dash__kpi-label">{label}</span>
      <strong className="dash__kpi-value">{value}</strong>
      {sub && <span className="dash__kpi-sub">{sub}</span>}
    </div>
  );

  const StageRows = ({ title, badge, items, empty, note }) => (
    <div className="dash__section">
      <div className="dash__section-head">
        <h3>{title}</h3>
        {items.length > 0 && (
          <span className={`dash__badge dash__badge--${badge}`}>{items.length}</span>
        )}
      </div>
      {note && <p className="dash__section-note">{note}</p>}
      {items.length === 0 ? (
        <p className="cards__empty">{empty}</p>
      ) : (
        <div className="dash__rows">
          {items.map((item) => (
            <article className="dash__row" key={item.id}>
              <div className="dash__row-main">
                <h4>{item.mreid || `Token #${item.id}`}</h4>
                <p className="dash__row-meta">
                  {item.stage || "Stage not reported"}
                  {item.isListed ? " · listed" : " · unlisted"}
                </p>
                <p className="dash__row-meta">
                  <span className="dash__row-eth">
                    {(() => {
                      try {
                        return `${Number(
                          ethers.utils.formatEther(item.escrowAmt || 0),
                        ).toFixed(2)} test ETH in escrow`;
                      } catch {
                        return "escrow amount not reported";
                      }
                    })()}
                  </span>
                  {` · sale price ${(() => {
                    try {
                      return `${Number(
                        ethers.utils.formatEther(item.price || 0),
                      ).toFixed(2)} test ETH`;
                    } catch {
                      return "not reported";
                    }
                  })()}`}
                </p>
                <p className="dash__row-meta">
                  {item.isListed ? "Buyer" : "Owner"} {short(
                    item.isListed ? item.buyer : item.owner,
                  )}
                  {item.inspectionPassed ? " · inspection passed" : ""}
                </p>
              </div>
              <div className="dash__row-side">
                <span className={`dash__tag dash__tag--${badge}`}>{badge}</span>
                {item.mreid ? (
                  <button
                    type="button"
                    className="dash__row-open"
                    onClick={() => onSelectProperty(item.mreid)}
                  >
                    Open property →
                  </button>
                ) : (
                  <span className="dash__row-unavailable">
                    MREID not available
                  </span>
                )}
              </div>
            </article>
          ))}
        </div>
      )}
    </div>
  );

  const portfolio = holdings && holdings.portfolio ? holdings.portfolio : null;
  const holdingsTotal = holdings ? holdings.count : 0;
  const holdingsPages = Math.max(Math.ceil(holdingsTotal / HOLDINGS_PAGE), 1);

  // ---------------------------------------------------------
  // WHICH ACTIVITY SECTIONS BELONG TO THIS WALLET
  //
  // The dashboard always leads with the two sections every visitor cares
  // about — the marketplace overview and the market intelligence — and then
  // shows the activity that matches the connected account:
  //
  //   seller    sold, properties for sale
  //   buyer     sold, purchases in escrow, owned through escrow
  //   inspector properties inspected
  //   lender    loans funded
  //
  // An account can be both a buyer and a seller, and an inspector or lender
  // can also own property, so the sections are composed rather than switched
  // between.  "My properties" and "Favourites" come last because they are the
  // same for every wallet.
  // ---------------------------------------------------------

  const showSeller = flags.seller || selling.length > 0 || sold.length > 0;
  const showBuyer = flags.buyer || buying.length > 0 || owned.length > 0;
  const isServiceRole = role === "Inspector" || role === "Lender";

  // A plain wallet gets the full buyer view even before it has any activity,
  // so the sections explain themselves.  An inspector or lender only sees the
  // owner sections once the chain shows they actually own or trade property.
  const showOwnerSections = isServiceRole ? showSeller || showBuyer : true;
  const showBuyingSections = showOwnerSections && (showBuyer || !showSeller);

  const activitySections = [
    role === "Inspector" ? (
      <StageRows
        key="inspected"
        title="Properties inspected"
        badge="inspected"
        items={inspected}
        empty="No inspections completed yet."
        note="Inspections recorded by this wallet on MillowEscrow."
      />
    ) : null,

    role === "Lender" ? (
      <StageRows
        key="lended"
        title="Loans funded"
        badge="lended"
        items={lended}
        empty="No loans funded yet."
        note="Amounts are denominated in test ETH for this local chain."
      />
    ) : null,

    showOwnerSections ? (
      <StageRows
        key="sold"
        title="Sold"
        badge="sold"
        items={sold}
        empty="No property has sold yet."
        note="Sales of this wallet that settled on chain. A settled sale moves the token to the buyer and cannot be undone."
      />
    ) : null,

    showOwnerSections && showSeller ? (
      <StageRows
        key="selling"
        title="Properties for sale"
        badge="selling"
        items={selling}
        empty="You have not listed any property yet. Use “List your home” in the marketplace to start a listing."
        note="Listings created by this wallet, read live from MillowEscrow. The listing, its terms and the escrow all live on the property page."
      />
    ) : null,

    showBuyingSections ? (
      <StageRows
        key="buying"
        title="Purchases in escrow"
        badge="buying"
        items={buying}
        empty="No purchase is in escrow right now."
        note="Each purchase shows how much has been committed and what is still required before the sale can settle."
      />
    ) : null,

    showBuyingSections ? (
      <StageRows
        key="owned"
        title="Owned through escrow"
        badge="owned"
        items={owned}
        empty="No completed purchase on this chain yet."
        note="Purchases of this wallet that settled, so the token now belongs to it."
      />
    ) : null,
  ].filter(Boolean);

  const overviewSection = (
    <div className="dash__section" key="overview">
      <div className="dash__section-head">
        <h3>Marketplace overview</h3>
      </div>

      {overviewError ? (
        <p className="cards__empty">
          Market overview unavailable: {overviewError}
        </p>
      ) : !overview ? (
        <p className="cards__empty">Loading market overview…</p>
      ) : (
        <>
          <div className="dash__kpis">
            <Kpi
              label="Properties listed"
              value={(overview.catalogue.total || 0).toLocaleString("en-IN")}
              sub="MREID catalogue"
            />
            <Kpi
              label="Tokenized"
              value={
                overview.catalogue.tokenized === null ||
                overview.catalogue.tokenized === undefined
                  ? "Not available"
                  : overview.catalogue.tokenized.toLocaleString("en-IN")
              }
              sub={
                overview.catalogue.chain && overview.catalogue.chain.available
                  ? `snapshot ${String(
                      overview.catalogue.chain.exported_at || "",
                    ).slice(0, 10)}`
                  : "chain snapshot unavailable"
              }
            />
            <Kpi
              label="On sale now"
              value={
                overview.catalogue.listed === null ||
                overview.catalogue.listed === undefined
                  ? "Not available"
                  : overview.catalogue.listed
              }
              sub="listed for sale on chain"
            />
            <Kpi
              label="Avg listed price"
              value={formatInrCompact(overview.ai.avg_listed)}
              sub={`avg AI estimate ${formatInrCompact(
                overview.ai.avg_ai_estimate,
              )}`}
            />
            <Kpi
              label="Potentially undervalued"
              value={overview.ai.undervalued.toLocaleString("en-IN")}
              sub={`${overview.ai.in_range.toLocaleString(
                "en-IN",
              )} near estimate`}
            />
            <Kpi
              label="Potentially overvalued"
              value={overview.ai.overvalued.toLocaleString("en-IN")}
              sub={
                overview.ai.model_mae_inr
                  ? `model MAE ${formatInrCompact(overview.ai.model_mae_inr)}`
                  : "model error not reported"
              }
            />
          </div>
          <p className="dash__section-note">{overview.ai.note}</p>
        </>
      )}
    </div>
  );

  const holdingsSection = (
    <div className="dash__section" key="holdings">
      <div className="dash__section-head">
        <h3>My properties</h3>
        {holdings && holdings.available && holdingsTotal > 0 && (
          <span className="dash__badge dash__badge--owned">{holdingsTotal}</span>
        )}
      </div>

      {portfolio && (
        <div className="dash__kpis dash__kpis--compact">
          <Kpi
            label="Properties held"
            value={portfolio.count.toLocaleString("en-IN")}
            sub={`${portfolio.on_sale_count} on sale`}
          />
          <Kpi
            label="Listed value"
            value={portfolio.listed_value_formatted}
            sub="sum of catalogue prices"
          />
          <Kpi
            label="MILLOW AI estimate"
            value={portfolio.ai_estimated_value_formatted}
            sub="research estimate"
          />
        </div>
      )}

      {holdingsError ? (
        <p className="cards__empty">{holdingsError}</p>
      ) : holdingsLoading && !holdings ? (
        <p className="cards__empty">Loading your properties…</p>
      ) : holdings && holdings.count === 0 ? (
        <p className="cards__empty">
          This wallet does not hold any property token yet. Buy a property in
          the marketplace and its NFT will appear here.
        </p>
      ) : holdings ? (
        <>
          <div className="dash__rows">
            {holdings.holdings.map((holding) => (
              <article className="dash__row" key={holding.mreid_id}>
                <div className="dash__row-main">
                  <h4>
                    {holding.summary
                      ? `${holding.summary.location}, ${holding.summary.city}`
                      : holding.mreid_id}
                  </h4>
                  <p className="dash__row-meta">
                    {holding.chain.token_id
                      ? `${holding.mreid_id} · token #${holding.chain.token_id}`
                      : holding.mreid_id}
                  </p>
                  {holding.summary ? (
                    <p className="dash__row-meta">
                      Listed {holding.summary.price_formatted} · AI{" "}
                      {holding.summary.ai_estimated_price_formatted} ·{" "}
                      {holding.summary.ai_market_signal.label}
                    </p>
                  ) : (
                    <p className="dash__row-meta">
                      {holding.note || "No catalogue data for this token."}
                    </p>
                  )}
                </div>
                <div className="dash__row-side">
                  <span
                    className={`dash__tag dash__tag--${
                      holding.chain.listed || holding.chain.active_sale
                        ? "selling"
                        : "owned"
                    }`}
                  >
                    {holding.chain.listed || holding.chain.active_sale
                      ? "for sale"
                      : "held"}
                  </span>
                  <button
                    type="button"
                    className="dash__row-open"
                    onClick={() => onSelectProperty(holding.mreid_id)}
                  >
                    Open property →
                  </button>
                </div>
              </article>
            ))}
          </div>

          {holdingsPages > 1 && (
            <nav className="mkt__pagination" aria-label="Holdings pages">
              <button
                type="button"
                className="mkt__page-btn"
                disabled={holdingsPage === 0 || holdingsLoading}
                onClick={() => setHoldingsPage((p) => Math.max(p - 1, 0))}
              >
                ← Prev
              </button>
              <span className="mkt__page-info">
                {holdingsLoading
                  ? "Loading…"
                  : `Page ${holdingsPage + 1} of ${holdingsPages}`}
              </span>
              <button
                type="button"
                className="mkt__page-btn"
                disabled={holdingsPage + 1 >= holdingsPages || holdingsLoading}
                onClick={() => setHoldingsPage((p) => p + 1)}
              >
                Next →
              </button>
            </nav>
          )}

          {holdings.note && <p className="dash__section-note">{holdings.note}</p>}
        </>
      ) : null}
    </div>
  );

  const favouritesSection = (
    <div className="dash__section" key="favourites">
      <div className="dash__section-head">
        <h3>Favourites</h3>
        {favorites.length > 0 && (
          <span className="dash__badge dash__badge--history">{favorites.length}</span>
        )}
      </div>

      {!favorites.length ? (
        <p className="cards__empty">
          No favourites yet. Save a listing from the marketplace with the heart
          on its card.
        </p>
      ) : favLoading && !favItems.length ? (
        <p className="cards__empty">Loading your favourites…</p>
      ) : (
        <>
          <div className="mkt__grid">
            {favItems.map((property) => (
              <PropertyCard
                key={property.mreid_id}
                property={property}
                onSelect={onSelectProperty}
                favorited
                onToggleFavorite={onToggleFavorite}
              />
            ))}
          </div>
          {favItems.length < favorites.length && (
            <p className="cards__empty">
              {favorites.length - favItems.length} favourite
              {favorites.length - favItems.length === 1 ? "" : "s"} could not be
              loaded from the registry.
            </p>
          )}
        </>
      )}
    </div>
  );

  const activitySection = error ? (
    <div className="dash__error" key="activity">
      <p>{error}</p>

      <button type="button" className="dash__retry" onClick={retry}>
        Retry
      </button>
    </div>
  ) : loading ? (
    <div className="dash__loading" key="activity">
      <div className="spinner"></div>

      <p>Loading your on-chain activity…</p>
    </div>
  ) : (
    <div key="activity">{activitySections}</div>
  );

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

      {overviewSection}

      <MarketIntelligence />

      {activitySection}

      {holdingsSection}

      {favouritesSection}
    </div>
  );
};

export default Dashboard;
