import { useEffect, useRef, useState } from "react";
import { cities, locations, propertiesByIds, searchProperties } from "../lib/millowApi";
import ListYourHome from "./ListYourHome";
import PropertyCard from "./PropertyCard";

const PAGE_SIZES = [12, 24, 48];
const SORT_OPTIONS = [
  { value: "id", label: "Sort: Featured" },
  { value: "price_asc", label: "Price: Low to High" },
  { value: "price_desc", label: "Price: High to Low" },
  { value: "area_asc", label: "Area: Small to Large" },
  { value: "area_desc", label: "Area: Large to Small" },
  { value: "price_per_sqft_asc", label: "Price/sqft: Low to High" },
  { value: "price_per_sqft_desc", label: "Price/sqft: High to Low" },
];

const EMPTY = "empty";

const FilterSelect = ({ value, onChange, options, placeholder, disabled }) => (
  <select
    className="mkt__select"
    value={value}
    onChange={onChange}
    disabled={disabled}
    aria-label={placeholder}
  >
    <option value={EMPTY}>{placeholder}</option>
    {options.map((opt) =>
      typeof opt === "string" ? (
        <option key={opt} value={opt}>
          {opt}
        </option>
      ) : (
        <option key={opt.value} value={opt.value}>
          {opt.label}
        </option>
      ),
    )}
  </select>
);

const NumberInput = ({ value, onChange, placeholder, label }) => (
  <input
    type="number"
    min="0"
    step="any"
    className="mkt__input"
    placeholder={placeholder}
    value={value}
    onChange={onChange}
    aria-label={label}
  />
);

const RentPanel = ({ setSection }) => (
  <section className="mkt__placeholder-panel">
    <div className="mkt__placeholder-icon" aria-hidden="true">
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.6">
        <path d="M12 3v18M3 12h18" />
      </svg>
    </div>
    <h2>Rentals are not in the registry yet</h2>
    <p>
      The MILLOW catalogue currently holds sale listings only — the on-chain
      sale contract has no rental terms, so there is no rental data to show
      here. Everything that is available is on the buy tab.
    </p>
    <button type="button" className="mkt__retry" onClick={() => setSection("buy")}>
      Browse sale listings
    </button>
  </section>
);

const SellPanel = ({ account, realEstate, onSelectProperty }) => (
  <main className="mkt">
    <ListYourHome
      account={account}
      realEstate={realEstate}
      onSelectProperty={onSelectProperty}
    />
  </main>
);

const Favorites = ({ favorites, onToggleFavorite, onSelectProperty, onShowAll }) => {
  const [items, setItems] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    propertiesByIds(favorites).then((rows) => {
      if (cancelled) return;
      setItems(rows);
      setLoading(false);
    });
    return () => {
      cancelled = true;
    };
  }, [favorites]);

  const missing = favorites.filter(
    (id) => !items.some((row) => row.mreid_id === id),
  );

  if (!favorites.length) {
    return (
      <section className="mkt__state mkt__empty-panel">
        <div className="mkt__empty-icon">♡</div>
        <h3>No favourites yet</h3>
        <p>
          Tap the heart on any listing to keep it here. Your favourites are
          stored per wallet on this device.
        </p>
        <button type="button" className="mkt__retry" onClick={onShowAll}>
          Browse properties
        </button>
      </section>
    );
  }

  return (
    <>
      {loading && (
        <section className="mkt__state">
          <div className="spinner" />
          <p>Loading your favourites…</p>
        </section>
      )}

      {!loading && items.length > 0 && (
        <section className="mkt__grid" aria-label="Favourite properties">
          {items.map((property) => (
            <PropertyCard
              key={property.mreid_id}
              property={property}
              onSelect={onSelectProperty}
              favorited
              onToggleFavorite={onToggleFavorite}
            />
          ))}
        </section>
      )}

      {!loading && missing.length > 0 && (
        <p className="mkt__hint">
          {missing.length} favourite{missing.length === 1 ? "" : "s"} could not
          be loaded from the registry: {missing.join(", ")}.
        </p>
      )}
    </>
  );
};

const Marketplace = ({
  section,
  setSection,
  favorites = [],
  onToggleFavorite,
  onSelectProperty,
  account,
  realEstate,
}) => {
  const [cityList, setCityList] = useState([]);
  const [locationOptions, setLocationOptions] = useState([]);

  const [q, setQ] = useState("");
  const [debouncedQ, setDebouncedQ] = useState("");
  const [city, setCity] = useState(EMPTY);
  const [selectedLocation, setSelectedLocation] = useState(EMPTY);
  const [minPrice, setMinPrice] = useState("");
  const [maxPrice, setMaxPrice] = useState("");
  const [minArea, setMinArea] = useState("");
  const [maxArea, setMaxArea] = useState("");
  const [bedrooms, setBedrooms] = useState(EMPTY);
  const [sort, setSort] = useState("id");
  const [page, setPage] = useState(1);
  const [pageSize, setPageSize] = useState(12);
  const [favOnly, setFavOnly] = useState(false);

  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [attempt, setAttempt] = useState(0);

  const controllerRef = useRef(null);

  useEffect(() => {
    cities()
      .then(setCityList)
      .catch(() => {});
  }, []);

  useEffect(() => {
    if (!city || city === EMPTY) {
      setLocationOptions([]);
      setSelectedLocation(EMPTY);
      return;
    }
    let cancelled = false;
    setLocationOptions([]);
    setSelectedLocation(EMPTY);
    locations(city)
      .then((list) => {
        if (!cancelled) setLocationOptions(list);
      })
      .catch(() => {});
    return () => {
      cancelled = true;
    };
  }, [city]);

  useEffect(() => {
    const timer = setTimeout(() => setDebouncedQ(q), 350);
    return () => clearTimeout(timer);
  }, [q]);

  const isBuy = section === "buy";
  const apiLocation = (debouncedQ || "").trim() || (selectedLocation !== EMPTY ? selectedLocation : "");

  useEffect(() => {
    if (!isBuy) return undefined;
    if (controllerRef.current) controllerRef.current.abort();
    const controller = new AbortController();
    controllerRef.current = controller;
    setLoading(true);
    setError(null);

    const params = {
      location: apiLocation || undefined,
      city: city !== EMPTY ? city : undefined,
      min_price: minPrice !== "" ? Number(minPrice) : undefined,
      max_price: maxPrice !== "" ? Number(maxPrice) : undefined,
      min_area: minArea !== "" ? Number(minArea) : undefined,
      max_area: maxArea !== "" ? Number(maxArea) : undefined,
      bedrooms: bedrooms !== EMPTY ? Number(bedrooms) : undefined,
      sort,
      page,
      page_size: pageSize,
    };

    searchProperties(params)
      .then((result) => {
        if (controller.signal.aborted) return;
        setData(result);
        setLoading(false);
      })
      .catch((err) => {
        if (controller.signal.aborted) return;
        setError(err.message);
        setLoading(false);
      });

    return () => controller.abort();
  }, [
    isBuy,
    apiLocation,
    city,
    minPrice,
    maxPrice,
    minArea,
    maxArea,
    bedrooms,
    sort,
    page,
    pageSize,
    attempt,
  ]);

  const resetPage = () => setPage(1);

  if (!isBuy) {
    return section === "sell" ? (
      <SellPanel
        account={account}
        realEstate={realEstate}
        onSelectProperty={onSelectProperty}
      />
    ) : (
      <RentPanel setSection={setSection} />
    );
  }

  const totalPages = data ? Math.max(data.total_pages, 1) : 1;
  const shownEmpty = !loading && !error && data && data.results.length === 0;

  const clearAll = () => {
    setQ("");
    setCity(EMPTY);
    setSelectedLocation(EMPTY);
    setMinPrice("");
    setMaxPrice("");
    setMinArea("");
    setMaxArea("");
    setBedrooms(EMPTY);
    setSort("id");
    resetPage();
  };

  return (
    <main className="mkt">
      <div className="mkt__hero">
        <h1>Find your next home in India</h1>
        <p>
          Browse {data ? data.total.toLocaleString("en-IN") : "thousands of"} buyers-market
          listings across Bangalore, Chennai, Delhi, Hyderabad, Kolkata and Mumbai.
        </p>
        <div className="mkt__searchbar">
          <svg viewBox="0 0 24 24" aria-hidden="true" className="mkt__search-icon">
            <circle cx="11" cy="11" r="7" />
            <line x1="21" y1="21" x2="16.5" y2="16.5" />
          </svg>
          <input
            type="text"
            className="mkt__search-input"
            placeholder="Search by locality or area name (e.g. Whitefield, JP Nagar)"
            value={q}
            onChange={(e) => {
              setQ(e.target.value);
              resetPage();
            }}
            aria-label="Search properties"
          />
          {q && (
            <button
              type="button"
              className="mkt__search-clear"
              aria-label="Clear search"
              onClick={() => setQ("")}
            >
              ×
            </button>
          )}
        </div>
        <div className="mkt__hero-actions">
          <button
            type="button"
            className={`mkt__chip-btn ${favOnly ? "mkt__chip-btn--on" : ""}`}
            onClick={() => setFavOnly((v) => !v)}
            aria-pressed={favOnly}
          >
            ♡ Favourites{favorites.length ? ` (${favorites.length})` : ""}
          </button>
          <button
            type="button"
            className="mkt__chip-btn"
            onClick={() => setSection("sell")}
          >
            + List your home
          </button>
        </div>
      </div>

      {favOnly ? (
        <Favorites
          favorites={favorites}
          onToggleFavorite={onToggleFavorite}
          onSelectProperty={onSelectProperty}
          onShowAll={() => setFavOnly(false)}
        />
      ) : (
        <>
          <section className="mkt__filters" aria-label="Property filters">
            <FilterSelect
              placeholder="City"
              value={city}
              onChange={(e) => {
                setCity(e.target.value);
                resetPage();
              }}
              options={cityList}
            />
            <FilterSelect
              placeholder="Location"
              value={selectedLocation}
              onChange={(e) => {
                setSelectedLocation(e.target.value);
                resetPage();
              }}
              options={locationOptions}
              disabled={!city || city === EMPTY}
            />
            <NumberInput
              label="Minimum price (₹)"
              placeholder="Min price"
              value={minPrice}
              onChange={(e) => {
                setMinPrice(e.target.value);
                resetPage();
              }}
            />
            <NumberInput
              label="Maximum price (₹)"
              placeholder="Max price"
              value={maxPrice}
              onChange={(e) => {
                setMaxPrice(e.target.value);
                resetPage();
              }}
            />
            <NumberInput
              label="Minimum area (sqft)"
              placeholder="Min area"
              value={minArea}
              onChange={(e) => {
                setMinArea(e.target.value);
                resetPage();
              }}
            />
            <NumberInput
              label="Maximum area (sqft)"
              placeholder="Max area"
              value={maxArea}
              onChange={(e) => {
                setMaxArea(e.target.value);
                resetPage();
              }}
            />
            <FilterSelect
              placeholder="Bedrooms"
              value={bedrooms}
              onChange={(e) => {
                setBedrooms(e.target.value);
                resetPage();
              }}
              options={["1", "2", "3", "4", "5", "6", "7", "8", "9"]}
            />
            <FilterSelect
              placeholder="Sort"
              value={sort}
              onChange={(e) => {
                setSort(e.target.value);
                resetPage();
              }}
              options={SORT_OPTIONS}
            />
          </section>

          <div className="mkt__meta">
            <span>
              {data && !error ? `${data.total.toLocaleString("en-IN")} properties found` : ""}
            </span>
            <span>
              {!error && data && (
                <>
                  Page {data.page} of {Math.max(data.total_pages, 1)}
                </>
              )}
            </span>
          </div>

          {error && (
            <section className="mkt__state mkt__error-panel">
              <div className="mkt__error-icon">!</div>
              <h3>Could not load properties</h3>
              <p className="mkt__error-msg">{error}</p>
              <p className="mkt__hint">
                Start the backend from the project root with <code>npm run valuation</code>.
              </p>
              <button
                type="button"
                className="mkt__retry"
                onClick={() => setAttempt((a) => a + 1)}
              >
                Retry
              </button>
            </section>
          )}

          {!error && shownEmpty && (
            <section className="mkt__state mkt__empty-panel">
              <div className="mkt__empty-icon">⌂</div>
              <h3>No properties match your filters</h3>
              <p>Try widening the price or area range, choosing another city, or clearing the search.</p>
              <button type="button" className="mkt__retry" onClick={clearAll}>
                Clear all filters
              </button>
            </section>
          )}

          {!error && !data && loading && (
            <section className="mkt__state">
              <div className="spinner" />
              <p>Loading properties…</p>
            </section>
          )}

          {!error && data && data.results.length > 0 && (
            <>
              <section className="mkt__grid" aria-label="Property listings">
                {data.results.map((property) => (
                  <PropertyCard
                    key={property.mreid_id}
                    property={property}
                    onSelect={onSelectProperty}
                    favorited={favorites.includes(property.mreid_id)}
                    onToggleFavorite={onToggleFavorite}
                  />
                ))}
              </section>

              <nav className="mkt__pagination" aria-label="Pagination">
                <button
                  type="button"
                  className="mkt__page-btn"
                  disabled={page <= 1 || loading}
                  onClick={() => setPage((p) => Math.max(p - 1, 1))}
                >
                  ← Prev
                </button>
                <span className="mkt__page-info">
                  {loading ? "Loading…" : `Page ${data.page} of ${totalPages}`}
                </span>
                <button
                  type="button"
                  className="mkt__page-btn"
                  disabled={page >= totalPages || loading}
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
                  {PAGE_SIZES.map((size) => (
                    <option key={size} value={size}>
                      {size} / page
                    </option>
                  ))}
                </select>
              </nav>
            </>
          )}
        </>
      )}

      <p className="mkt__disclaimer">
        Listed prices come from the MREID dataset. MILLOW AI values are research
        estimates, not certified appraisals.
      </p>
    </main>
  );
};

export default Marketplace;
