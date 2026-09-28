import { useEffect, useState } from "react";
import { propertyChain, searchProperties } from "../lib/millowApi";

const short = (address) =>
  address ? `${address.slice(0, 6)}…${address.slice(-4)}` : "";

// The MILLOW listing flow.  A property can only be listed by the wallet that
// holds its on-chain NFT, so this panel resolves the property in the registry,
// verifies ownership against the live chain (falling back to the exported
// snapshot when no wallet is connected) and then hands the user over to the
// property page, where the actual listing transaction lives.
const ListYourHome = ({ account, realEstate, onSelectProperty }) => {
  const [query, setQuery] = useState("");
  const [debounced, setDebounced] = useState("");
  const [results, setResults] = useState([]);
  const [searching, setSearching] = useState(false);
  const [searchError, setSearchError] = useState(null);
  const [checking, setChecking] = useState(null);
  const [verdict, setVerdict] = useState(null);

  useEffect(() => {
    const timer = setTimeout(() => setDebounced(query.trim()), 350);
    return () => clearTimeout(timer);
  }, [query]);

  useEffect(() => {
    if (!debounced) {
      setResults([]);
      return undefined;
    }
    let cancelled = false;
    setSearching(true);
    setSearchError(null);
    searchProperties({ location: debounced, page_size: 6 })
      .then((data) => {
        if (!cancelled) setResults(data.results || []);
      })
      .catch((err) => {
        if (!cancelled) setSearchError(err.message);
      })
      .finally(() => {
        if (!cancelled) setSearching(false);
      });
    return () => {
      cancelled = true;
    };
  }, [debounced]);

  const verify = async (property) => {
    setVerdict(null);
    setChecking(property.mreid_id);
    try {
      const payload = await propertyChain(property.mreid_id);
      const snapshot = (payload && payload.chain) || {};
      if (!snapshot.tokenized || !snapshot.token_id) {
        setVerdict({
          mreid: property.mreid_id,
          state: "not-tokenized",
          snapshot,
        });
        return;
      }

      // A live read is authoritative when a wallet is connected; otherwise the
      // exported snapshot is the best available answer and is labelled as such.
      let owner = snapshot.owner || null;
      let source = "snapshot";
      if (realEstate && snapshot.token_id) {
        try {
          owner = await realEstate.ownerOf(snapshot.token_id);
          source = "chain";
        } catch (err) {
          console.warn("ownerOf failed, falling back to the snapshot", err);
        }
      }

      setVerdict({
        mreid: property.mreid_id,
        state: account && owner && owner.toLowerCase() === account.toLowerCase()
          ? "owned"
          : "not-owned",
        owner,
        source,
        tokenId: snapshot.token_id,
        snapshotAt: (payload.chain_snapshot || {}).exported_at,
      });
    } catch (err) {
      setVerdict({ mreid: property.mreid_id, state: "error", message: err.message });
    } finally {
      setChecking(null);
    }
  };

  return (
    <section className="sell" aria-label="List your property">
      <div className="sell__head">
        <h2>List your property on MILLOW</h2>
        <p>
          MILLOW lists a property as its on-chain NFT. Find the property in the
          registry, confirm your wallet holds it, then set the sale terms — the
          listing, the escrow and the transfer all happen on the property page.
        </p>
      </div>

      <ol className="sell__steps">
        <li>
          <strong>Find it</strong>
          <span>Search the MILLOW registry by locality or area name.</span>
        </li>
        <li>
          <strong>Verify it</strong>
          <span>
            We check that your connected wallet holds the property token.
          </span>
        </li>
        <li>
          <strong>List it</strong>
          <span>Set price, escrow and buyer terms, then sign the listing.</span>
        </li>
      </ol>

      {!account && (
        <p className="sell__notice">
          Connect your wallet to list a property. You can browse the
          marketplace without connecting.
        </p>
      )}

      <div className="sell__search">
        <input
          type="text"
          className="sell__input"
          placeholder="Search your locality (e.g. Whitefield, Kokapilly)"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          aria-label="Search your property in the MILLOW registry"
        />
        {query && (
          <button
            type="button"
            className="mkt__search-clear"
            aria-label="Clear"
            onClick={() => setQuery("")}
          >
            ×
          </button>
        )}
      </div>

      {searching && <p className="sell__hint">Searching the registry…</p>}
      {searchError && <p className="sell__notice">{searchError}</p>}

      {!searching && debounced && results.length === 0 && !searchError && (
        <p className="sell__hint">
          No property in the MILLOW registry matches “{debounced}”.
        </p>
      )}

      <ul className="sell__results">
        {results.map((property) => (
          <li key={property.mreid_id} className="sell__result">
            <div>
              <span className="mkt__card-city">{property.city}</span>
              <h3 className="mkt__card-location">{property.location}</h3>
              <p className="mkt__card-meta">
                <strong>{property.bedrooms}</strong> bds ·{" "}
                <strong>{Number(property.area).toLocaleString("en-IN")}</strong>{" "}
                sqft · {property.price_formatted}
              </p>
            </div>
            <button
              type="button"
              className="sell__check"
              disabled={checking === property.mreid_id}
              onClick={() => verify(property)}
            >
              {checking === property.mreid_id ? "Checking…" : "Check ownership"}
            </button>
          </li>
        ))}
      </ul>

      {verdict && (
        <div className={`sell__verdict sell__verdict--${verdict.state}`} role="status">
          {verdict.state === "owned" && (
            <>
              <strong>You own this property.</strong>{" "}
              Token #{verdict.tokenId}
              {verdict.source === "snapshot" && " (chain snapshot)"}. Open the
              property to set the sale terms and sign the listing.
              <button
                type="button"
                className="sell__primary"
                onClick={() => onSelectProperty(verdict.mreid)}
              >
                List {verdict.mreid}
              </button>
            </>
          )}
          {verdict.state === "not-owned" && (
            <>
              <strong>Your wallet does not hold this property.</strong> The
              token is owned by {short(verdict.owner) || "another address"}
              {verdict.source === "snapshot" &&
                ` (chain snapshot${verdict.snapshotAt ? ` from ${verdict.snapshotAt.slice(0, 10)}` : ""})`}
              . MILLOW only lists a property from the wallet that owns it — ask
              the owner to transfer the token to you first.
            </>
          )}
          {verdict.state === "not-tokenized" && (
            <>
              <strong>{verdict.mreid} is not tokenized on this chain.</strong>{" "}
              There is nothing to list until the property has an NFT.
            </>
          )}
          {verdict.state === "error" && (
            <>
              <strong>Could not check ownership.</strong> {verdict.message}
            </>
          )}
        </div>
      )}
    </section>
  );
};

export default ListYourHome;
