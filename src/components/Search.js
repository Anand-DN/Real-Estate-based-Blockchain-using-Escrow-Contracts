const Search = ({ search, setSearch }) => {
    return (
        <header>
            <h2 className="header__title">Search it. Explore it. Buy it.</h2>
            <input
                type="text"
                className="header__search"
                placeholder="Enter an address, neighborhood, city, or ZIP code"
                value={search}
                onChange={(e) => setSearch(e.target.value)}
            />
        </header>
    );
}

export default Search;