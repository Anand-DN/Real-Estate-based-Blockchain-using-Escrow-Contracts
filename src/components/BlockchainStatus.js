import { useEffect, useState } from "react";
import { getPropertyChainStatus, demoRoleOf, ROLE_NAMES } from "../lib/blockchain";

const LEGAL =
  "Digital property representation. Blockchain-based ownership record for the prototype. Technical proof-of-concept. Legal ownership remains subject to applicable Indian property and registration law.";

const shortOf = (value) =>
  value ? `${String(value).slice(0, 10)}…${String(value).slice(-6)}` : null;

const Field = ({ label, children }) => (
  <div className="chain__field">
    <span className="mkt__label">{label}</span>
    <span className="chain__value">{children}</span>
  </div>
);

const CopyButton = ({ value, label }) => {
  const [copied, setCopied] = useState(false);
  const copy = async () => {
    try {
      if (navigator.clipboard && navigator.clipboard.writeText) {
        await navigator.clipboard.writeText(value);
        setCopied(true);
      }
    } catch (err) {
      setCopied(false);
    }
  };
  return (
    <button
      type="button"
      className="chain__copy"
      onClick={copy}
      aria-label={`Copy ${label}`}
    >
      {copied ? "Copied" : "Copy"}
    </button>
  );
};

const BlockchainStatus = ({
  mreidId,
  chainRecord,
  refreshTick,
}) => {
  const [status, setStatus] = useState(null);
  const [loading, setLoading] = useState(true);
  const [attempt, setAttempt] = useState(0);

  useEffect(() => {
    if (!mreidId) return undefined;
    let cancelled = false;
    setLoading(true);
    getPropertyChainStatus(mreidId, chainRecord)
      .then((result) => {
        if (!cancelled) setStatus(result);
      })
      .catch((err) => {
        if (!cancelled)
          setStatus({
            available: false,
            reason: "error",
            error: String(err && err.message ? err.message : err),
          });
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [mreidId, chainRecord, attempt, refreshTick]);

  const ready = Boolean(status && status.available);

  return (
    <section className="chain">
      <h3 className="mkt__section-title">Blockchain &amp; NFT Status</h3>

      {loading && (
        <div className="mkt__nhb-loading">
          <div className="spinner" />
          <p>Checking the local blockchain…</p>
        </div>
      )}

      {!loading && !ready && status && (
        <div className="mkt__nhb-error">
          <p>Blockchain unavailable.</p>
          <p className="mkt__error-msg">
            {status.error || "Could not reach the local Hardhat node."}
          </p>
          <button
            type="button"
            className="mkt__retry"
            onClick={() => setAttempt((a) => a + 1)}
          >
            Retry
          </button>
        </div>
      )}

      {!loading && ready && status.reason === "no-contracts" && (
        <div className="mkt__nhb-error">
          <p>No MILLOW contracts configured on chain {status.chainId}.</p>
          <p className="mkt__error-msg">
            {status.error ||
              "Run scripts/deployMillow.js then scripts/mintMreid.js against the local node."}
          </p>
        </div>
      )}

      {!loading &&
        ready &&
        !status.supported &&
        status.reason !== "no-contracts" && (
          <div className="mkt__nhb-error">
            <p>Unsupported network.</p>
            <p className="mkt__error-msg">
              This prototype reads the local development chain (chain 31337).
              The connected node reports chain {status.chainId} (
              {status.chainName}).
            </p>
          </div>
        )}

      {!loading && ready && status.supported && !status.tokenizationUnknown && (
        <div className="chain__grid">
          <Field label="Network">{status.chainName}</Field>
          <Field label="Chain ID">{status.chainId}</Field>
          <Field label="Property NFT contract">
            {shortOf(status.nftAddress)}
          </Field>
          <Field label="MREID / property ID">{status.mreidId}</Field>

          {status.registered ? (
            <>
              <Field label="Token ID">#{status.tokenId}</Field>
              <Field label="Owner">
                {shortOf(status.owner)}
                {demoRoleOf(status.owner) && (
                  <span> ({ROLE_NAMES[demoRoleOf(status.owner)]})</span>
                )}
              </Field>
              <Field label="Listing status">
                <span
                  className={`chain__chip ${
                    status.finalized
                      ? "chain__chip--inactive"
                      : status.onOffer
                        ? "chain__chip--active"
                        : "chain__chip--inactive"
                  }`}
                >
                  {status.finalized
                    ? "Sold"
                    : status.onOffer === null
                      ? "Unavailable"
                      : status.onOffer
                        ? "On offer"
                        : "Not listed"}
                </span>
              </Field>
              <Field label="Metadata URI">{status.tokenURI}</Field>
              <Field label="Mint transaction">
                <span>{status.mintTxHash ? shortOf(status.mintTxHash) : "—"}</span>
                {status.mintTxHash && (
                  <CopyButton value={status.mintTxHash} label="mint transaction" />
                )}
              </Field>
              {status.mintBlock && (
                <Field label="Mint block">#{status.mintBlock}</Field>
              )}
            </>
          ) : (
            <>
              <Field label="Token ID">Not registered</Field>
              <Field label="Owner">—</Field>
              <Field label="Listing status">—</Field>
            </>
          )}
        </div>
      )}

      {!loading && status && status.supported && status.tokenizationUnknown && (
        <div className="tx-card">
          <h4>Token status unavailable</h4>
          <p className="tx-hint">
            The MILLOW index and the connected blockchain do not currently
            provide a confirmed token for this MREID. This is different from a
            confirmed missing token, so no tokenization or transaction status
            has been fabricated.
          </p>
          {status.tokenId ? <p className="tx-hint">Indexed token ID: #{status.tokenId}</p> : null}
          {status.error ? <p className="tx-hint tx-warn">{status.error}</p> : null}
        </div>
      )}

      {!loading && status && status.supported && status.registered === false && (
        <div className="tx-card">
          <h4>No NFT yet</h4>
          <p className="tx-hint">
            This MREID record has no token on the prototype chain, so it has no
            on-chain ownership, listing, escrow or history. The simplified
            MILLOW workflow operates on properties that are already on the
            chain with their NFT at the seller.
          </p>
        </div>
      )}

      {!loading && status && status.supported && status.registered && status.workflowAvailable === false && (
        <div className="tx-card">
          <h4>Read-only chain state</h4>
          <p className="tx-hint">
            The NFT is present on the connected chain, but the MILLOW escrow
            deployment required for transactions is unavailable. Listing and
            ownership values above are read from the connected contracts.
          </p>
        </div>
      )}

      {!loading && (
        <div className="mkt__disclaimer chain__legal">
          <strong>About this status</strong>
          <p>{LEGAL}</p>
        </div>
      )}
    </section>
  );
};

export default BlockchainStatus;