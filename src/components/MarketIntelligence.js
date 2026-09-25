import { useEffect, useMemo, useRef, useState } from "react";
import {
  cities,
  dashboardInsights,
  dashboardOverview,
  marketBreakdown,
  searchProperties,
} from "../lib/millowApi";
import { formatInrCompact } from "../lib/format";
import PropertyDetail from "./PropertyDetail";

const EMPTY = "empty";

const SIGNAL_OPTIONS = [
  { value: "undervalued", label: "Potential buy (undervalued)" },
  { value: "overvalued", label: "Potentially overvalued" },
  { value: "in_range", label: "Near AI estimate" },
];

const SORT_OPTIONS = [
  { value: "id", label: "Sort: Featured" },
  { value: "price_asc", label: "Listed price: Low to High" },
  { value: "price_desc", label: "Listed price: High to Low" },
  { value: "area_asc", label: "Area: Small to Large" },
  { value: "area_desc", label: "Area: Large to Small" },
  { value: "price_per_sqft_asc", label: "Price/sqft: Low to High" },
  { value: "price_per_sqft_desc", label: "Price/sqft: High to Low" },
  { value: "ai_estimate_asc", label: "AI estimate: Low to High" },
  { value: "ai_estimate_desc", label: "AI estimate: High to Low" },
  { value: "ai_difference_asc", label: "AI gap: Largest overvalue" },
  { value: "ai_difference_desc", label: "AI gap: Largest undervalue" },
];

const Stat = ({ label, value, sub, tone }) => (
  <div className={`mi__stat${tone ? ` mi__stat--${tone}` : ""}`}>
    <span className="mi__stat-label">{label}</span>
    <strong className="mi__stat-value">{value}</strong>
    {sub && <span className="mi__stat-sub">{sub}</span>}
  </div>
);

const MarketIntelligence = () => {
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [insightList, setInsightList] = useState([]);
  const [loadError, setLoadError] = useState(null);
  const [metaLoading, setMetaLoading] = useState(true);

  const [cityList, setCityList] = useState([]);
  const [city, setCity] = useState(EMPTY);
  const [aiSignal, setAiSignal] = useState(EMPTY);
  const [sort, setSort] = useState("ai_difference_desc");
  const [page, setPage] = useState(1);
  const [pageSize, setPageSize] = useState(24);

  const [data, setData] = useState(null);
  const [tableLoading, setTableLoading] = useState(true);
  const [tableError, setTableError] = useState(null);
  const [attempt, setAttempt] = useState(0);
  const [selectedId, setSelectedId] = useState(null);

  const controllerRef = useRef(null);

  useEffect(() => {
    let cancelled = false;
    setMetaLoading(true);
    setLoadError(null);
    Promise.all([
      cities().catch(() => []),
      dashboardOverview().catch((err) => {
        throw err;
      }),
      marketBreakdown({ group: "city" }).catch((err) => {
        throw err;
      }),
      dashboardInsights().catch((err) => {
        throw err;
      }),
    ])
      .then(([citiesList, ov, bd, ins]) => {
        if (cancelled) return;
        setCityList(citiesList);
        setOverview(ov);
        setBreakdown(bd);
        setInsightList(Array.isArray(ins.insights) ? ins.insights : []);
      })
      .catch((err) => {
        if (!cancelled) setLoadError(err.message);
      })
      .finally(() => {
        if (!cancelled) setMetaLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    if (controllerRef.current) controllerRef.current.abort();
    const controller = new AbortController();
    controllerRef.current = controller;
    setTableLoading(true);
    setTableError(null);

    searchProperties({
      city: city !== EMPTY ? city : undefined,
      ai_signal: aiSignal !== EMPTY ? aiSignal : undefined,
      sort,
      page,
      page_size: pageSize,
    })
      .then((result) => {
        if (controller.signal.aborted) return;
        setData(result);
        setTableLoading(false);
      })
      .catch((err) => {
        if (controller.signal.aborted) return;
        setTableError(err.message);
        setTableLoading(false);
      });

    return () => controller.abort();
  }, [city, aiSignal, sort, page, pageSize, attempt]);

  const resetPage = () => setPage(1);
  const totalPages = data ? Math.max(data.total_pages, 1) : 1;

  const chain = overview?.catalogue?.chain;
  const ai = overview?.ai;

  // Simple horizontal bar scale for the city comparison chart.
  const barScale = useMemo(() => {
    if (!breakdown || !breakdown.rows || !breakdown.rows.length) return null;
    const maxListed = Math.max(
      ...breakdown.rows.map((r) => Number(r.avg_listed) || 0),
      1,
    );
    const maxAi = Math.max(
      ...breakdown.rows.map((r) => Number(r.avg_ai) || 0),
      1,
    );
    return { maxListed, maxAi };
  }, [breakdown]);

  const signalLabel = (label) => {
    if (!label) return "—";
    if (label.includes("undervalued"))
      return <span className="mi__signal mi__signal--buy">undervalued</span>;
    if (label.includes("overvalued"))
      return <span className="mi__signal mi__signal--sell">overvalued</span>;
    return <span className="mi__signal mi__signal--flat">in range</span>;
  };

  return (
    <section className="mi" aria-label="Market intelligence">
      <div className="mi__hero">
        <h2>Market Intelligence</h2>
        <p>
          Live analytics over the full MREID catalogue: tokenization state, the
          MILLOW V5 AI research estimate, and market-wide signals.
        </p>
      </div>

      {metaLoading ? (
        <div className="mi__loading">
          <div className="spinner" />
          <p>Loading market analytics…</p>
        </div>
      ) : loadError ? (
        <div className="mi__error" role="alert">
          <strong>Market analytics unavailable.</strong> {loadError}. Start the
          backend with <code>npm run valuation</code> and refresh.
        </div>
      ) : (
        overview && (
          <>
            {/* ---------------- KPI CARDS ---------------- */}
            <div className="mi__kpis">
              <Stat
                label="Catalogue"
                value={Number(chain?.total_properties ?? overview.catalogue.total ?? 0).toLocaleString("en-IN")}
                sub="properties indexed"
              />
              <Stat
                label="Tokenized"
                value={Number(chain?.tokenized ?? overview.catalogue.tokenized ?? 0).toLocaleString("en-IN")}
                tone="ok"
                sub={
                  chain
                    ? `on-chain ${chain.total_properties === chain.tokenized ? "complete" : ""}`
                    : "chain snapshot pending"
                }
              />
              <Stat
                label="Listed"
                value={String(chain?.listed ?? overview.catalogue.listed ?? 0)}
                tone="buy"
                sub={`${chain?.active_sales ?? overview.catalogue.active_sales ?? 0} active sales`}
              />
              <Stat
                label="AI undervalued"
                value={String(ai?.undervalued ?? 0).toLocaleString("en-IN")}
                tone="buy"
                sub="potential buy signals"
              />
              <Stat
                label="AI overvalued"
                value={String(ai?.overvalued ?? 0).toLocaleString("en-IN")}
                tone="sell"
                sub="above the AI estimate"
              />
              <Stat
                label="AI near estimate"
                value={String(ai?.in_range ?? 0).toLocaleString("en-IN")}
                sub="within ±5% of AI"
              />
              <Stat
                label="Avg listed"
                value={formatInrCompact(ai?.avg_listed)}
                sub={`≈ ${formatInrCompact(ai?.avg_listed_ppsf)}/sqft`}
              />
              <Stat
                label="Avg AI estimate"
                value={formatInrCompact(ai?.avg_ai_estimate)}
                sub={`model MAE ${formatInrCompact(ai?.model_mae_inr)}`}
              />
            </div>

            {ai?.note && <p className="mi__note">{ai.note}</p>}

            {/* ---------------- MARKET BREAKDOWN ---------------- */}
            {breakdown && breakdown.rows && breakdown.rows.length > 0 && (
              <div className="mi__block">
                <div className="mi__block-head">
                  <h3>Average price by city</h3>
                  <span>Listed vs MILLOW AI research estimate (₹)</span>
                </div>
                <div className="mi__chart">
                  {breakdown.rows.map((row) => (
                    <div className="mi__chart-row" key={row.city}>
                      <div className="mi__chart-label">
                        <strong>{row.city}</strong>
                        <span>
                          {Number(row.count).toLocaleString("en-IN")} props
                        </span>
                      </div>
                      <div className="mi__chart-bars">
                        <div className="mi__bar-line">
                          <span className="mi__bar-caption">Listed</span>
                          <div className="mi__bar-track">
                            <div
                              className="mi__bar mi__bar--listed"
                              style={{
                                width: `${Math.round(
                                  (row.avg_listed / barScale.maxListed) * 100,
                                )}%`,
                              }}
                            />
                          </div>
                          <span className="mi__bar-value">
                            {formatInrCompact(row.avg_listed)}
                          </span>
                        </div>
                        <div className="mi__bar-line">
                          <span className="mi__bar-caption">AI</span>
                          <div className="mi__bar-track">
                            <div
                              className="mi__bar mi__bar--ai"
                              style={{
                                width: `${Math.round(
                                  (row.avg_ai / barScale.maxAi) * 100,
                                )}%`,
                              }}
                            />
                          </div>
                          <span className="mi__bar-value">
                            {formatInrCompact(row.avg_ai)}
                          </span>
                        </div>
                      </div>
                      <div className="mi__chart-tags">
                        <span className="mi__tag mi__tag--buy">
                          {row.undervalued} ✓
                        </span>
                        <span className="mi__tag mi__tag--sell">
                          {row.overvalued} ✗
                        </span>
                      </div>
                    </div>
                  ))}
                </div>
                {breakdown.note && <p className="mi__note">{breakdown.note}</p>}
              </div>
            )}

            {/* ---------------- INSIGHTS ---------------- */}
            {insightList.length > 0 && (
              <div className="mi__block">
                <div className="mi__block-head">
                  <h3>Market insights</h3>
                  <span>Deterministic, computed from the real catalogue</span>
                </div>
                <ul className="mi__insights">
                  {insightList.map((item, index) => (
                    <li key={index}>
                      <span className={`mi__kind mi__kind--${item.kind === "DATA FACT" ? "fact" : "model"}`}>
                        {item.kind}
                      </span>
                      <span className="mi__insight-text">{item.text}</span>
                    </li>
                  ))}
                </ul>
                <p className="mi__note">
                  MODEL-BASED INTERPRETATION items are research signals, not
                  certified valuations or investment advice.
                </p>
              </div>
            )}
          </>
        )
      )}

      {/* ---------------- CATALOGUE TABLE ---------------- */}
      <div className="mi__block mi__catalogue">
        <div className="mi__block-head">
          <h3>AI-sorted catalogue</h3>
          <span>Filter by market signal and rank by the AI gap</span>
        </div>

        <div className="mi__filters">
          <select
            className="mkt__select"
            value={city}
            onChange={(e) => {
              setCity(e.target.value);
              resetPage();
            }}
            aria-label="Filter by city"
          >
            <option value={EMPTY}>All cities</option>
            {cityList.map((c) => (
              <option key={c} value={c}>
                {c}
              </option>
            ))}
          </select>

          <select
            className="mkt__select"
            value={aiSignal}
            onChange={(e) => {
              setAiSignal(e.target.value);
              resetPage();
            }}
            aria-label="Filter by AI market signal"
          >
            <option value={EMPTY}>All AI signals</option>
            {SIGNAL_OPTIONS.map((opt) => (
              <option key={opt.value} value={opt.value}>
                {opt.label}
              </option>
            ))}
          </select>

          <select
            className="mkt__select"
            value={sort}
            onChange={(e) => {
              setSort(e.target.value);
              resetPage();
            }}
            aria-label="Sort properties"
          >
            {SORT_OPTIONS.map((opt) => (
              <option key={opt.value} value={opt.value}>
                {opt.label}
              </option>
            ))}
          </select>
        </div>

        {tableError && (
          <div className="mi__error" role="alert">
            <strong>Could not load the catalogue.</strong> {tableError}
            <button
              type="button"
              className="mkt__retry"
              onClick={() => setAttempt((a) => a + 1)}
            >
              Retry
            </button>
          </div>
        )}

        {!tableError && tableLoading && !data && (
          <div className="mi__loading">
            <div className="spinner" />
            <p>Loading catalogue…</p>
          </div>
        )}

        {!tableError && data && (
          <>
            <div className="mi__meta">
              <span>
                {data.total.toLocaleString("en-IN")} properties · Page {data.page}{" "}
                of {totalPages}
              </span>
            </div>

            {data.results.length === 0 ? (
              <p className="cards__empty">
                No properties match those filters.
              </p>
            ) : (
              <div className="mi__table-wrap">
                <table className="mi__table">
                  <thead>
                    <tr>
                      <th>Property</th>
                      <th>Listed</th>
                      <th>AI estimate</th>
                      <th>Gap</th>
                      <th>Signal</th>
                    </tr>
                  </thead>
                  <tbody>
                    {data.results.map((row) => {
                      const signal = row.ai_market_signal || {};
                      const diff = signal.difference_pct;
                      const gapClass =
                        diff < 0
                          ? "mi__gap mi__gap--under"
                          : diff > 0
                            ? "mi__gap mi__gap--over"
                            : "";
                      return (
                        <tr
                          key={row.mreid_id}
                          onClick={() => setSelectedId(row.mreid_id)}
                          tabIndex={0}
                          onKeyDown={(e) => {
                            if (e.key === "Enter" || e.key === " ") {
                              e.preventDefault();
                              setSelectedId(row.mreid_id);
                            }
                          }}
                          aria-label={`${row.mreid_id}, ${row.location}, ${row.city} — listed ${row.price_formatted}`}
                        >
                          <td className="mi__cell-id">
                            <strong>{row.mreid_id}</strong>
                            <span>
                              {row.location}, {row.city}
                            </span>
                            <span className="mi__cell-sub">
                              {row.bedrooms} bds · {Number(row.area).toLocaleString("en-IN")} sqft
                            </span>
                          </td>
                          <td className="mi__cell-price">
                            {row.price_formatted}
                            <span className="mi__cell-sub">
                              {formatInrCompact(row.price_per_sqft)}/sqft
                            </span>
                          </td>
                          <td className="mi__cell-price">
                            {row.ai_estimated_price_formatted}
                            <span className="mi__cell-sub">
                              {formatInrCompact(row.ai_estimated_price_per_sqft)}/sqft
                            </span>
                          </td>
                          <td className="mi__cell-gap">
                            {typeof diff === "number" ? (
                              <span className={gapClass}>
                                {diff > 0 ? "+" : ""}
                                {Math.round(diff * 100) / 100}%
                              </span>
                            ) : (
                              "—"
                            )}
                          </td>
                          <td>{signalLabel(signal.label)}</td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            )}

            <nav className="mkt__pagination" aria-label="Catalogue pagination">
              <button
                type="button"
                className="mkt__page-btn"
                disabled={page <= 1 || tableLoading}
                onClick={() => setPage((p) => Math.max(p - 1, 1))}
              >
                ← Prev
              </button>
              <span className="mkt__page-info">
                {tableLoading ? "Loading…" : `Page ${data.page} of ${totalPages}`}
              </span>
              <button
                type="button"
                className="mkt__page-btn"
                disabled={page >= totalPages || tableLoading}
                onClick={() => setPage((p) => Math.min(p + 1, totalPages))}
              >
                Next →
              </button>
              <select
                className="mkt__page-size"
                aria-label="Results per page"
                value={pageSize}
                onChange={(e) => {
                  setPageSize(Number(e.target.value));
                  resetPage();
                }}
              >
                {[12, 24, 48].map((size) => (
                  <option key={size} value={size}>
                    {size} / page
                  </option>
                ))}
              </select>
            </nav>
          </>
        )}
      </div>

      {selectedId && (
        <PropertyDetail
          key={selectedId}
          mreidId={selectedId}
          onClose={() => setSelectedId(null)}
          onSelectSimilar={setSelectedId}
        />
      )}
    </section>
  );
};

export default MarketIntelligence;