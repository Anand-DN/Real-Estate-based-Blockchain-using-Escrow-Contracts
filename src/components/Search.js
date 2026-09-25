const Search = ({ search, setSearch }) => {
    return (
        <header>
            <h2 className="header__title">Search it. Explore it. Buy it.</h2>
            <div className="search__bar">
                <span className="search__icon" aria-hidden="true">
                    <svg
                        viewBox="0 0 24 24"
                        fill="none"
                        stroke="currentColor"
                        strokeWidth="2.2"
                        strokeLinecap="round"
                        strokeLinejoin="round"
                    >
                        <circle cx="11" cy="11" r="7" />
                        <line x1="21" y1="21" x2="16.5" y2="16.5" />
                    </svg>
                </span>
                <input
                    type="text"
                    className="header__search"
                    placeholder="Enter an address, neighborhood, city, or ZIP code"
                    value={search}
                    onChange={(e) => setSearch(e.target.value)}
                    aria-label="Search properties"
                />
                {search.length > 0 && (
                    <button
                        type="button"
                        className="search__clear"
                        aria-label="Clear search"
                        onClick={() => setSearch("")}
                    >
                        <svg
                            viewBox="0 0 24 24"
                            fill="none"
                            stroke="currentColor"
                            strokeWidth="2.4"
                            strokeLinecap="round"
                        >
                            <path d="M18 6 6 18M6 6l12 12" />
                        </svg>
                    </button>
                )}
            </div>
        </header>
    );
}

export default Search;