import { ethers } from 'ethers';
import logo from '../assets/logo.svg';

const Navigation = ({ account, setAccount, view, setView, setShowList, theme, onToggleTheme }) => {
    const connectHandler = async () => {
        const accounts = await window.ethereum.request({ method: 'eth_requestAccounts' });
        const account = ethers.utils.getAddress(accounts[0])
        setAccount(account);
    }

    return (
        <nav>
            <ul className='nav__links'>
                <li><button type="button" className="nav__link" onClick={() => setView('browse')}>Buy</button></li>
                <li><button type="button" className="nav__link" onClick={() => setView('browse')}>Rent</button></li>
                <li><button type="button" className="nav__link" onClick={() => setShowList(true)}>Sell</button></li>
                <li>
                    <button type="button" className="nav__btn" onClick={() => setView(view === 'dashboard' ? 'browse' : 'dashboard')}>
                        {view === 'dashboard' ? 'Browse' : 'Dashboard'}
                    </button>
                </li>
            </ul>

            <div className='nav__brand'>
                <img src={logo} alt="Logo" />
                <h1>Millow</h1>
            </div>

            <div className='nav__right'>
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
                    >
                        {account.slice(0, 6) + '...' + account.slice(38, 42)}
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