import { useEffect, useState } from "react";
import { marketContext, propertyById, recommendations, riskAnalysis } from "../lib/millowApi";
import { formatInr } from "../lib/format";
import PropertyThumb from "./PropertyThumb";
import PropertyCard from "./PropertyCard";
import BlockchainStatus from "./BlockchainStatus";
import TransactionPanel from "./TransactionPanel";

const prettify = (key) =>
  String(key)
    .replace(/([a-z0-9])([A-Z])/g, "$1 $2")
    .replace(/_/g, " ")
    .replace(/\s+/g, " ")
    .trim()
    .replace(/\b\w/g, (c) => c.toUpperCase());

const PROPERTY_FACTS = [
  { key: "area", label: "Area", format: (v) => `${Number(v).toLocaleString("en-IN")} sqft` },
  { key: "bedrooms", label: "Bedrooms", format: (v) => String(v) },
  { key: "resale", label: "Resale", format: (v) => (Number(v) ? "Yes" : "No") },
];

const PropertyDetail = ({ mreidId, onClose, onSelectSimilar }) => {
  const [detail, setDetail] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [attempt, setAttempt] = useState(0);

  const [market, setMarket] = useState(null);
  const [marketLoading, setMarketLoading] = useState(false);
  const [marketError, setMarketError] = useState(null);
  const [marketAttempt, setMarketAttempt] = useState(0);

  const [risk, setRisk] = useState(null);
  const [riskLoading, setRiskLoading] = useState(false);
  const [riskError, setRiskError] = useState(null);
  const [riskAttempt, setRiskAttempt] = useState(0);

  const [recs, setRecs] = useState(null);
  const [recsLoading, setRecsLoading] = useState(false);
  const [recsError, setRecsError] = useState(null);
  const [recsAttempt, setRecsAttempt] = useState(0);
  const [saleTick, setSaleTick] = useState(0);

  useEffect(() => {
    if (!mreidId) return;
    let cancelled = false;
    setLoading(true);
    setError(null);
    propertyById(mreidId)
      .then((data) => {
        if (!cancelled) setDetail(data);
      })
      .catch((err) => {
        if (!cancelled) setError(err.message);
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [mreidId, attempt]);

  useEffect(() => {
    if (!mreidId) return;
    let cancelled = false;
    setMarketLoading(true);
    setMarketError(null);
    marketContext(mreidId)
      .then((data) => {
        if (!cancelled) setMarket(data);
      })
      .catch((err) => {
        if (!cancelled) setMarketError(err.message);
      })
      .finally(() => {
        if (!cancelled) setMarketLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [mreidId, marketAttempt]);

  useEffect(() => {
    if (!mreidId) return;
    let cancelled = false;
    setRiskLoading(true);
    setRiskError(null);
    riskAnalysis(mreidId)
      .then((data) => {
        if (!cancelled) setRisk(data);
      })
      .catch((err) => {
        if (!cancelled) setRiskError(err.message);
      })
      .finally(() => {
        if (!cancelled) setRiskLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [mreidId, riskAttempt]);

  useEffect(() => {
    if (!mreidId) return;
    let cancelled = false;
    setRecsLoading(true);
    setRecsError(null);
    recommendations(mreidId, { limit: 6 })
      .then((data) => {
        if (!cancelled) setRecs(data);
      })
      .catch((err) => {
        if (!cancelled) setRecsError(err.message);
      })
      .finally(() => {
        if (!cancelled) setRecsLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [mreidId, recsAttempt]);

  useEffect(() => {
    if (!mreidId) return undefined;
    const onKey = (e) => {
      if (e.key === "Escape") onClose();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [mreidId, onClose]);

  return (
    <div
      className="mkt__detail-backdrop"
      onClick={onClose}
      role="presentation"
    >
      <div
        className="mkt__detail"
        role="dialog"
        aria-modal="true"
        aria-label="Property details"
        onClick={(e) => e.stopPropagation()}
      >
        <button
          type="button"
          className="mkt__detail-close"
          aria-label="Close details"
          onClick={onClose}
        >
          ×
        </button>

        {loading && (
          <div className="mkt__detail-center">
            <div className="spinner" />
            <p>Loading property details…</p>
          </div>
        )}

        {!loading && error && (
          <div className="mkt__detail-center">
            <div className="mkt__error-icon">!</div>
            <p>Could not load property details.</p>
            <p className="mkt__error-msg">{error}</p>
            <button
              type="button"
              className="mkt__retry"
              onClick={() => setAttempt((a) => a + 1)}
            >
              Retry
            </button>
          </div>
        )}

        {!loading && !error && detail && (
          <div key={mreidId} className="mkt__detail-content">
            <PropertyThumb mreidId={detail.mreid_id} city={detail.property.city} />

            <div className="mkt__detail-body">
              <span className="mkt__card-city">{detail.property.city}</span>
              <h2 className="mkt__detail-location">{detail.property.location}</h2>

              <div className="mkt__detail-facts">
                {PROPERTY_FACTS.map((f) => (
                  <div className="mkt__fact" key={f.key}>
                    <span className="mkt__label">{f.label}</span>
                    <strong>{f.format(detail.property[f.key])}</strong>
                  </div>
                ))}
              </div>

              <div className="mkt__detail-prices">
                <div className="mkt__detail-listed">
                  <span className="mkt__label">Listed price</span>
                  <strong className="mkt__detail-listed-value">
                    {detail.listed_price_formatted}
                  </strong>
                  <span className="mkt__detail-listed-sqft">
                    {formatInr(detail.price_per_sqft)}/sqft
                  </span>
                </div>
                <div className="mkt__detail-ai-panel">
                  <span className="mkt__label">
                    MILLOW AI estimate <span className="mkt__ai-badge">AI</span>
                  </span>
                  <strong className="mkt__detail-ai-value">
                    {detail.ai_estimation.ai_estimated_price_formatted}
                  </strong>
                  <span className="mkt__detail-ai-sqft">
                    ≈ {formatInr(detail.ai_estimation.ai_estimated_price_per_sqft)}/sqft
                  </span>
                  <span className="mkt__detail-ai-model">
                    {detail.ai_estimation.model}
                  </span>
                </div>
              </div>

              {detail.ai_estimation.note && (
                <p className="mkt__detail-note">{detail.ai_estimation.note}</p>
              )}

              <h3 className="mkt__section-title">NHB Market Context</h3>

              {marketLoading && (
                <div className="mkt__nhb-loading">
                  <div className="spinner" />
                  <p>Loading city-level NHB RESIDEX benchmark…</p>
                </div>
              )}

              {!marketLoading && marketError && (
                <div className="mkt__nhb-error">
                  <p>Could not load NHB market context.</p>
                  <p className="mkt__error-msg">{marketError}</p>
                  <button
                    type="button"
                    className="mkt__retry"
                    onClick={() => setMarketAttempt((a) => a + 1)}
                  >
                    Retry
                  </button>
                </div>
              )}

              {!marketLoading && !marketError && market && (
                <div className="mkt__nhb">
                  <div className="mkt__nhb-badge">
                    External benchmark - city-level
                  </div>

                  <div className="mkt__nhb-grid">
                    <div className="mkt__fact">
                      <span className="mkt__label">Benchmark city</span>
                      <strong>
                        {market.market_context.city} (NHB RESIDEX)
                      </strong>
                    </div>
                    <div className="mkt__fact">
                      <span className="mkt__label">Observation quarter</span>
                      <strong>{market.market_context.quarter}</strong>
                    </div>
                    <div className="mkt__fact">
                      <span className="mkt__label">Composite benchmark</span>
                      <strong>
                        {formatInr(
                          market.market_context.composite_price_rs_sqft,
                        )}
                        /sqft
                      </strong>
                    </div>
                    {market.market_context.applicable_size_band && (
                      <div className="mkt__fact">
                        <span className="mkt__label">
                          Size band (
                          {market.market_context.applicable_size_band.label})
                        </span>
                        <strong>
                          {formatInr(
                            market.market_context.applicable_size_band
                              .price_rs_sqft,
                          )}
                          /sqft
                        </strong>
                      </div>
                    )}
                    <div className="mkt__fact">
                      <span className="mkt__label">HPI assessment</span>
                      <strong>{market.market_context.hpi_assessment}</strong>
                    </div>
                    <div className="mkt__fact mkt__fact--wide">
                      <span className="mkt__label">Series / period</span>
                      <strong>
                        {market.market_context.series} -{" "}
                        {market.market_context.access_period}
                      </strong>
                    </div>
                    {market.market_context.source_url && (
                      <div className="mkt__fact mkt__fact--wide">
                        <span className="mkt__label">Source</span>
                        <strong>
                          {market.market_context.source}{" "}
                          <a
                            href={market.market_context.source_url}
                            target="_blank"
                            rel="noopener noreferrer"
                            className="mkt__nhb-source"
                          >
                            {market.market_context.source_url}
                          </a>
                        </strong>
                      </div>
                    )}
                  </div>

                  <div className="mkt__compare">
                    <div className="mkt__compare-title">
                      Reference figures - each measures something different
                    </div>
                    <div className="mkt__compare-row">
                      <span className="mkt__compare-label">
                        {market.city} listed price
                      </span>
                      <span className="mkt__compare-value">
                        {formatInr(market.property.listed_price_per_sqft)}/sqft
                      </span>
                    </div>
                    <div className="mkt__compare-row">
                      <span className="mkt__compare-label">
                        MILLOW AI estimate
                      </span>
                      <span className="mkt__compare-value">
                        {formatInr(
                          market.millow_ai.ai_estimated_price_per_sqft,
                        )}
                        /sqft
                      </span>
                    </div>
                    <div className="mkt__compare-row">
                      <span className="mkt__compare-label">
                        NHB city benchmark
                      </span>
                      <span className="mkt__compare-value">
                        {formatInr(
                          market.market_context.composite_price_rs_sqft,
                        )}
                        /sqft
                      </span>
                      <span className="mkt__compare-note">
                        city-level index, not this property
                      </span>
                    </div>
                  </div>

                  <p className="mkt__nhb-note">{market.market_context.note}</p>
                </div>
              )}

              <h3 className="mkt__section-title">
                Transaction Risk & Anomaly Analysis
              </h3>

              {riskLoading && (
                <div className="mkt__nhb-loading">
                  <div className="spinner" />
                  <p>Loading anomaly analysis…</p>
                </div>
              )}

              {!riskLoading && riskError && (
                <div className="mkt__nhb-error">
                  <p>Could not load the anomaly analysis.</p>
                  <p className="mkt__error-msg">{riskError}</p>
                  <button
                    type="button"
                    className="mkt__retry"
                    onClick={() => setRiskAttempt((a) => a + 1)}
                  >
                    Retry
                  </button>
                </div>
              )}

              {!riskLoading && !riskError && risk && risk.analysis_subject && (
                <div className="mkt__risk">
                  <div className="mkt__risk-score">
                    <div className="mkt__risk-score-value">
                      {risk.anomaly_score}
                      <span className="mkt__risk-score-max">/100</span>
                    </div>
                    <div className="mkt__risk-score-side">
                      <div className="mkt__risk-score-title">
                        Anomaly score
                      </div>
                      <div className="mkt__risk-score-note">
                        How unusual this listing is vs comparable MREID
                        listings. Not a probability.
                      </div>
                    </div>
                  </div>

                  {Array.isArray(risk.indicators) &&
                    risk.indicators.length > 0 && (
                      <ul className="mkt__risk-list">
                        {(risk.indicators || []).map((ind) => (
                          <li
                            className="mkt__risk-item"
                            key={ind.id || ind.title}
                          >
                            <div className="mkt__risk-item-head">
                              <span className="mkt__risk-item-title">
                                {ind.title}
                              </span>
                              <span
                                className={`mkt__chip mkt__chip--${ind.status || "none"}`}
                              >
                                {ind.status || "none"}
                              </span>
                            </div>
                            {ind.z !== null &&
                              ind.z !== undefined &&
                              ind.z !== "n/a" && (
                                <div className="mkt__risk-item-z">
                                  robust z: {ind.z.toString()} ·{" "}
                                  {ind.direction === "above_typical"
                                    ? "above typical"
                                    : ind.direction === "below_typical"
                                    ? "below typical"
                                    : ind.direction}
                                </div>
                              )}
                            <p className="mkt__risk-item-expl">
                              {ind.explanation}
                            </p>
                          </li>
                        ))}
                      </ul>
                    )}

                  {risk.horizon_context && (
                    <div className="mkt__risk-context">
                      <span className="mkt__risk-context-label">
                        City-level transaction context ({risk.horizon_context.city})
                      </span>
                      <span>
                        {risk.horizon_context.n_transactions.toLocaleString(
                          "en-IN",
                        )}{" "}
                        transactions
                      </span>
                      <span>
                        median ₹/sqft{" "}
                        {formatInr(
                          risk.horizon_context.median_price_per_sqft,
                        )}
                      </span>
                      <span>
                        {risk.horizon_context.rapid_repeat_share * 100}% rapid
                        repeat sales
                      </span>
                      <span className="mkt__risk-context-note">
                        {risk.horizon_context.synthetic_note}
                      </span>
                    </div>
                  )}

                  <p className="mkt__risk-method">{risk.methodology}</p>

                  {risk.disclaimer && (
                    <div className="mkt__disclaimer">
                      <strong>About this analysis</strong>
                      <p>{risk.disclaimer}</p>
                    </div>
                  )}
                </div>
              )}

              <h3 className="mkt__section-title">Similar Properties</h3>

              {recsLoading && (
                <div className="mkt__nhb-loading">
                  <div className="spinner" />
                  <p>Finding similar properties…</p>
                </div>
              )}

              {!recsLoading && recsError && (
                <div className="mkt__nhb-error">
                  <p>Could not load similar properties.</p>
                  <p className="mkt__error-msg">{recsError}</p>
                  <button
                    type="button"
                    className="mkt__retry"
                    onClick={() => setRecsAttempt((a) => a + 1)}
                  >
                    Retry
                  </button>
                </div>
              )}

              {!recsLoading && !recsError && recs && (
                <div>
                  {recs.items && recs.items.length > 0 ? (
                    <div className="mkt__recs-grid">
                      {recs.items.map((item) => (
                        <article
                          className="mkt__rec"
                          key={item.mreid_id}
                        >
                          <div className="mkt__rec-score">
                            <span>{item.recommendation_score}</span>
                            <i>relevance</i>
                          </div>
                          <PropertyCard
                            property={item.property}
                            onSelect={
                              onSelectSimilar ||
                              (() => {})
                            }
                          />
                          <div className="mkt__rec-reasons">
                            {item.reasons.slice(0, 3).map((reason) => (
                              <span className="mkt__rec-reason" key={reason}>
                                {reason}
                              </span>
                            ))}
                          </div>
                        </article>
                      ))}
                    </div>
                  ) : (
                    <div className="mkt__nhb-loading">
                      <p>No similar properties found.</p>
                    </div>
                  )}
                </div>
              )}

              <BlockchainStatus
                mreidId={detail.mreid_id}
                chainRecord={detail.chain}
                refreshTick={saleTick}
              />

              <TransactionPanel
                mreidId={detail.mreid_id}
                chainRecord={detail.chain}
                listedPriceInr={detail.listed_price}
                onSettled={() => setSaleTick((t) => t + 1)}
                refreshTick={saleTick}
              />

              <h3 className="mkt__section-title">Amenities</h3>
              <div className="mkt__amenities">
                {Object.entries(detail.property.amenities || {}).map(
                  ([key, value]) => {
                    const state =
                      value === "Yes"
                        ? "yes"
                        : value === "No"
                        ? "no"
                        : "unknown";
                    return (
                      <span
                        className={`mkt__amenity mkt__amenity--${state}`}
                        key={key}
                      >
                        {prettify(key)}
                        <i>{value}</i>
                      </span>
                    );
                  },
                )}
              </div>

              <div className="mkt__disclaimer">
                <strong>About this estimate</strong>
                <p>{detail.disclaimer}</p>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default PropertyDetail;