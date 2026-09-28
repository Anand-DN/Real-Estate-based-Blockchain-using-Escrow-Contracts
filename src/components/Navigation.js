import { ethers } from 'ethers';
import logo from '../assets/logo.svg';

const shortAddress = (address) =>
    `${address.slice(0, 6)}...${address.slice(38, 42)}`;

const Navigation = ({
    account,
    setAccount,
    view,
    section,
    onGoMarketplace,
    onGoDashboard,
    onGoHome,
    theme,
    onToggleTheme,
}) => {
    const connectHandler = async () => {
        try {
            const accounts = await window.ethereum.request({ method: 'eth_requestAccounts' });
            setAccount(ethers.utils.getAddress(accounts[0]));
        } catch (error) {
            // User rejected the request or no wallet present; keep the button idle
            console.error('Could not connect wallet', error)
        }
    }

    const onMarketplace = view === 'marketplace';
    const link = (target) =>
        `nav__link ${onMarketplace && section === target ? 'nav__link--active' : ''}`;

    return (
        <nav>
            <ul className='nav__links'>
                <li>
                    <button
                        type="button"
                        className={link('buy')}
                        onClick={() => onGoMarketplace('buy')}
                    >
                        Buy
                    </button>
                </li>
                <li>
                    <button
                        type="button"
                        className={link('rent')}
                        onClick={() => onGoMarketplace('rent')}
                    >
                        Rent
                    </button>
                </li>
                <li>
                    <button
                        type="button"
                        className={link('sell')}
                        onClick={() => onGoMarketplace('sell')}
                    >
                        Sell
                    </button>
                </li>
            </ul>

            <button
                type="button"
                className='nav__brand'
                onClick={onGoHome}
                aria-label="MILLOW home"
                title="MILLOW home"
            >
                <img src={logo} alt="Logo" />
                <h1>Millow</h1>
            </button>

            <div className='nav__right'>
                <button
                    type="button"
                    className={`nav__btn ${onMarketplace ? 'nav__btn--active' : ''}`}
                    onClick={() => onGoMarketplace('buy')}
                    aria-current={onMarketplace ? 'page' : undefined}
                >
                    Marketplace
                </button>
                <button
                    type="button"
                    className={`nav__btn ${view === 'dashboard' ? 'nav__btn--active' : ''}`}
                    onClick={onGoDashboard}
                    aria-current={view === 'dashboard' ? 'page' : undefined}
                >
                    Dashboard
                </button>
                <button
                    type="button"
                    className="theme__toggle"
                    title={theme === 'dark' ? 'Switch to light theme' : 'Switch to dark theme'}
                    onClick={onToggleTheme}
                >
                    {theme === 'dark' ? (
                        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
                            <circle cx="12" cy="12" r="4" />
                            <path d="M12 2v2m0 16v2M4.93 4.93l1.41 1.41m11.32 11.32 1.41 1.41M2 12h2m16 0h2M4.93 19.07l1.41-1.41m11.32-11.32 1.41-1.41" />
                        </svg>
                    ) : (
                        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                            <path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z" />
                        </svg>
                    )}
                </button>

                {account ? (
                    <button
                        type="button"
                        className='nav__connect'
                        title={account}
                    >
                        {shortAddress(account)}
                    </button>
                ) : (
                    <button
                        type="button"
                        className='nav__connect'
                        onClick={connectHandler}
                    >
                        Connect
                    </button>
                )}
            </div>
        </nav>
    );
}

export default Navigation;
