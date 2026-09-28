import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { ethers } from 'ethers';
import App from './App';

jest.setTimeout(60000);

// These run against the real services: the MILLOW chain on :8545 and the
// property/valuation API on :8001.  They assert that the landing page is the
// real marketplace and that the dashboard is built from real catalogue and
// chain-snapshot data.
describe('Marketplace live integration', () => {
  let node;

  beforeAll(() => {
    node = new ethers.providers.JsonRpcProvider('http://localhost:8545');
    window.ethereum = {
      request: (args) => node.send(args.method, args.params || []),
      on: () => {},
      removeListener: () => {},
    };
  });

  // jsdom keeps the URL between tests, and the app routes from it.
  beforeEach(() => {
    window.history.pushState({}, '', '/');
  });

  it('lands on the real MREID marketplace', async () => {
    const accounts = await node.send('eth_accounts', []);
    expect(accounts.length).toBeGreaterThan(0);

    render(<App />);

    expect(
      screen.getByRole('heading', { name: /find your next home in india/i }),
    ).toBeInTheDocument();

    // Real listings from the catalogue, not placeholders.  The first request
    // warms the valuation model in the backend, so allow for it.
    const cards = await screen.findAllByRole(
      'button',
      { name: /listed ₹/i },
      { timeout: 30000 },
    );
    expect(cards.length).toBeGreaterThan(0);
    expect(screen.getAllByText(/properties found/i).length).toBeGreaterThan(0);
    expect(screen.getAllByText(/MILLOW AI estimate/i).length).toBe(cards.length);
  });

  it('builds the dashboard from real catalogue and wallet data', async () => {
    const accounts = await node.send('eth_accounts', []);
    const account = ethers.utils.getAddress(accounts[0]);

    render(<App />);
    await screen.findAllByRole('button', { name: /listed ₹/i }, { timeout: 30000 });

    fireEvent.click(screen.getByRole('button', { name: 'Dashboard' }));

    await waitFor(
      () => expect(screen.getByText('My Dashboard')).toBeInTheDocument(),
      { timeout: 30000 },
    );
    await waitFor(
      () =>
        expect(
          screen.queryByText('Loading your on-chain activity…'),
        ).not.toBeInTheDocument(),
      { timeout: 30000 },
    );

    expect(
      screen.getAllByText(account.slice(0, 6) + '...' + account.slice(38, 42))
        .length,
    ).toBeGreaterThanOrEqual(1);

    // The two sections every wallet gets come first, in this order, and the
    // wallet-wide sections come last.  The activity sections in between depend
    // on what the chain says this account has done, so only their position is
    // asserted, not their presence.
    const before = (a, b) =>
      Boolean(a.compareDocumentPosition(b) & Node.DOCUMENT_POSITION_FOLLOWING);
    const overview = screen.getByRole('heading', { name: 'Marketplace overview' });
    const intelligence = screen.getByRole('heading', { name: 'Market Intelligence' });
    const mine = screen.getByRole('heading', { name: 'My properties' });
    const favourites = screen.getByRole('heading', { name: 'Favourites' });
    expect(before(overview, intelligence)).toBe(true);
    expect(before(intelligence, mine)).toBe(true);
    expect(before(mine, favourites)).toBe(true);

    for (const title of ['Properties for sale', 'Purchases in escrow', 'Sold', 'Owned through escrow']) {
      const section = screen.queryByRole('heading', { name: title });
      if (section) {
        expect(before(intelligence, section)).toBe(true);
        expect(before(section, mine)).toBe(true);
      }
    }

    const sections = screen.getAllByText(
      /does not hold any property token yet|Loading your properties|holdings|You have not listed any property yet|unavailable|No property has sold yet/i,
    );
    expect(sections.length).toBeGreaterThan(0);
  });

  it('opens a real property page from a marketplace card', async () => {
    render(<App />);

    const [card] = await screen.findAllByRole(
      'button',
      { name: /listed ₹/i },
      { timeout: 30000 },
    );
    const label = card.getAttribute('aria-label');
    fireEvent.click(card);

    const dialog = await screen.findByRole('dialog');
    expect(dialog).toBeInTheDocument();
    expect(label).toMatch(/listed/i);

    const mreid = window.location.pathname.split('/').pop();
    expect(mreid).toMatch(/^MREID_\d{7}$/);
  });
});
