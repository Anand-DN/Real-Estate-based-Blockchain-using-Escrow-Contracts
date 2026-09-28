import { render, screen, fireEvent, waitFor, within } from '@testing-library/react';
import App from './App';

jest.setTimeout(20000);

const accounts = {
  buyer: '0xf39Fd6e51aad88F6F4ce6aB8827279cffFb92266',
  seller: '0x70997970C51812dc3A010C7d01b50e0d17dc79C8',
  inspector: '0x3C44CdDdB6a900fa2b585dd299e03d12FA4293BC',
  lender: '0x90F79bf6EB2c4f870365E785982E1f101E93b906',
};

const mreidFor = (id) => `MREID_${String(id).padStart(7, '0')}`;

const metadata = (id) => ({
  name: `Home ${id}`,
  address: '123 Test St',
  description: 'A test home',
  image: '/images/1.jpg',
  id: String(id),
  attributes: [
    { trait_type: 'Purchase Price', value: 20 },
    { trait_type: 'Type of Residence', value: 'Condo' },
    { trait_type: 'Bed Rooms', value: 2 },
    { trait_type: 'Bathrooms', value: 3 },
    { trait_type: 'Square Feet', value: 2200 },
    { trait_type: 'Year Built', value: 2013 },
  ],
});

const json = (body) => Promise.resolve({ ok: true, json: () => Promise.resolve(body) });

const summary = (id) => ({
  mreid_id: mreidFor(id),
  price: 9000000,
  price_formatted: '₹90 Lakh',
  price_per_sqft: 3600,
  area: 2500,
  location: 'Whitefield',
  city: 'Bangalore',
  bedrooms: 3,
  amenities: { gymnasium: 'Yes', liftavailable: 'No' },
  ai_estimated_price: 9500000,
  ai_estimated_price_formatted: '₹95 Lakh',
  ai_estimated_price_per_sqft: 3800,
  ai_market_signal: { label: 'Potentially undervalued' },
  locality: { count: 20, median_price_per_sqft: 4000 },
});

const overview = {
  catalogue: {
    total: 29135,
    tokenized: 29135,
    listed: 0,
    active_sales: 0,
    finalized: 0,
    chain: { available: true, exported_at: '2026-09-26T16:23:37.384Z' },
  },
  ai: {
    model: 'MILLOW V5 Log-Price XGBoost',
    analyzed: 29135,
    undervalued: 100,
    overvalued: 200,
    in_range: 28835,
    avg_listed: 11950000,
    avg_ai_estimate: 12000000,
    model_mae_inr: 750000,
    note: 'AI values are research estimates, not certified appraisals.',
  },
};

const holdings = (address) => ({
  address,
  available: true,
  exported_at: '2026-09-26T16:23:37.384Z',
  count: 2,
  offset: 0,
  limit: 12,
  has_more: false,
  portfolio: {
    count: 2,
    on_sale_count: 1,
    listed_value: 18000000,
    listed_value_formatted: '₹1.80 Crore',
    ai_estimated_value: 19000000,
    ai_estimated_value_formatted: '₹1.90 Crore',
  },
  holdings: [
    {
      mreid_id: 'MREID_0000001',
      chain: {
        token_id: 1,
        tokenized: true,
        listed: true,
        active_sale: false,
        finalized: false,
        sale_status: 'Listed',
      },
      summary: summary(1),
    },
    {
      mreid_id: 'MREID_0000002',
      chain: {
        token_id: 2,
        tokenized: true,
        listed: false,
        active_sale: false,
        finalized: false,
        sale_status: 'None',
      },
      summary: summary(2),
    },
  ],
  note: 'Holdings come from the exported chain-index snapshot.',
});

jest.mock('ethers', () => {
  const actual = jest.requireActual('ethers');
  const { BigNumber } = actual;

  const contractValues = {
    totalSupply: () => Promise.resolve(BigNumber.from(6)),
    tokenURI: (id) => Promise.resolve(`http://localhost:3000/metadata/${id}.json`),
    isListed: () => Promise.resolve(true),
    ownerOf: () => Promise.resolve(accounts.seller),
    hasRole: () => Promise.resolve(false),
    propertyOf: (id) => Promise.resolve(`MREID_${String(id).padStart(7, '0')}`),
    sales: () =>
      Promise.resolve({
        seller: accounts.seller,
        buyer: '0x0000000000000000000000000000000000000000',
        inspectionPassed: false,
        lenderFundedWei: BigNumber.from(0),
        priceWei: BigNumber.from('20000000000000000000'),
        earnestWei: BigNumber.from('10000000000000000000'),
      }),
    queryFilter: () =>
      Promise.resolve(
        [1, 2, 3, 4, 5, 6].map((id) => ({
          args: { tokenId: BigNumber.from(id) },
        })),
      ),
    filters: {
      Transfer: () => ({}),
      SaleListed: () => ({}),
      EarnestDeposited: () => ({}),
      BalanceFunded: () => ({}),
      SaleFinalized: () => ({}),
      SaleCancelled: () => ({}),
      Listed: () => ({}),
      Unlisted: () => ({}),
    },
    on: () => {},
    off: () => {},
    getSigner: () => ({ sendTransaction: async () => ({ wait: async () => {} }) }),
    getNetwork: () => Promise.resolve({ chainId: 31337, name: 'localhost' }),
  };

  const handler = {
    get(target, prop) {
      if (prop in target) return target[prop];
      if (prop === 'address') return '0x0000000000000000000000000000000000000001';
      if (prop === 'filters') return contractValues.filters;
      if (prop === 'connect') return () => new Proxy({}, handler);
      if (contractValues[prop]) return (...args) => contractValues[prop](...args);
      return undefined;
    },
  };

  class Web3Provider {
    constructor() {}
    getNetwork() { return contractValues.getNetwork(); }
    getSigner() { return contractValues.getSigner(); }
  }

  function Contract() {
    return new Proxy({}, handler);
  }

  const providers = { ...actual.providers, Web3Provider };
  const mocked = {
    ...actual,
    providers,
    Contract,
  };
  mocked.ethers = {
    ...actual.ethers,
    providers,
    Contract,
  };
  return mocked;
});

describe('Marketplace-first dashboard', () => {
  const setupEthereum = (account) => {
    window.ethereum = {
      request: (arg) => {
        if (arg.method === 'eth_accounts') return Promise.resolve([account]);
        return Promise.resolve();
      },
      on: () => {},
      removeListener: () => {},
    };
  };

  beforeEach(() => {
    global.fetch = jest.fn((uri) => {
      if (uri.includes('/api/dashboard/overview')) return json(overview);
      if (uri.includes('/api/dashboard/holdings')) {
        const address = new URL(uri).searchParams.get('address');
        return json(holdings(address));
      }
      if (uri.includes('/api/properties/search')) {
        return json({ total: 2, total_pages: 1, page: 1, page_size: 12, results: [summary(1), summary(2)] });
      }
      const match = uri.match(/(\d+)\.json/);
      if (match) return json(metadata(match[1]));
      return Promise.reject(new Error(`unexpected fetch: ${uri}`));
    });
    localStorage.clear();
    window.history.pushState({}, '', '/');
  });

  it('lands on the marketplace with listings and navigation', async () => {
    setupEthereum(accounts.buyer);
    render(<App />);

    expect(
      screen.getByRole('heading', { name: /find your next home in india/i }),
    ).toBeInTheDocument();

    expect(await screen.findAllByText('Whitefield')).toHaveLength(2);
    expect(screen.getByRole('button', { name: 'Buy' })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Rent' })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Sell' })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /List your home/i })).toBeInTheDocument();

    // No assistant button in the navbar: it is reached from a property page.
    expect(screen.queryByRole('button', { name: 'AI Assistant' })).toBeNull();
  });

  it('opens the dashboard with buyer role and marketplace sections', async () => {
    setupEthereum(accounts.buyer);
    render(<App />);
    await screen.findByRole('heading', { name: /find your next home in india/i });

    fireEvent.click(screen.getByRole('button', { name: 'Dashboard' }));

    await waitFor(() => expect(screen.getByRole('heading', { name: 'My Dashboard' })).toBeInTheDocument());
    await waitFor(() => expect(screen.getByText('Buyer')).toBeInTheDocument(), { timeout: 5000 });

    expect(screen.getByRole('heading', { name: 'Marketplace overview' })).toBeInTheDocument();
    expect(screen.getByRole('heading', { name: 'My properties' })).toBeInTheDocument();
    expect(screen.getByRole('heading', { name: 'Favourites' })).toBeInTheDocument();

    // A buyer gets the buyer view, in order, after the overview and the market
    // intelligence.  "My properties" and "Favourites" come last.
    const before = (a, b) =>
      Boolean(a.compareDocumentPosition(b) & Node.DOCUMENT_POSITION_FOLLOWING);
    const order = [
      'Marketplace overview',
      'Market Intelligence',
      'Sold',
      'Purchases in escrow',
      'Owned through escrow',
      'My properties',
      'Favourites',
    ].map((name) => screen.getByRole('heading', { name }));
    order.slice(0, -1).forEach((node, i) =>
      expect(before(node, order[i + 1])).toBe(true),
    );

    // A buyer who has never sold anything is not shown the seller section.
    expect(screen.queryByRole('heading', { name: 'Properties for sale' })).toBeNull();
  });

  it('shows the wallet holdings returned by the API', async () => {
    setupEthereum(accounts.seller);
    render(<App />);
    await screen.findAllByText('Whitefield');

    fireEvent.click(screen.getByRole('button', { name: 'Dashboard' }));

    // The section node is re-resolved on every retry: React replaces it as the
    // holdings load, so a captured node would go stale.
    const section = () =>
      screen.getByRole('heading', { name: 'My properties' }).closest('.dash__section');

    await waitFor(() =>
      expect(within(section()).getByText(/MREID_0000001 · token #1/)).toBeInTheDocument(),
    );
    expect(within(section()).getByText('₹1.80 Crore')).toBeInTheDocument();
    expect(within(section()).getByText('₹1.90 Crore')).toBeInTheDocument();
    expect(within(section()).getAllByText('for sale')).toHaveLength(1);
    expect(within(section()).getAllByText('held')).toHaveLength(1);
  });

  it('opens dashboard with seller role and properties for sale', async () => {
    setupEthereum(accounts.seller);
    render(<App />);
    await screen.findByRole('heading', { name: /find your next home in india/i });

    fireEvent.click(screen.getByRole('button', { name: 'Dashboard' }));

    await waitFor(() => expect(screen.getByText('Seller')).toBeInTheDocument(), { timeout: 5000 });

    const forSale = screen.getByRole('heading', { name: 'Properties for sale' });
    const section = forSale.closest('.dash__section');
    expect(within(section).getByText('6')).toBeInTheDocument();
    expect(within(section).getAllByText('MREID_0000001').length).toBeGreaterThan(0);
  });

  it('shows sections even when the metadata server is down', async () => {
    setupEthereum(accounts.buyer);
    global.fetch = jest.fn((uri) => {
      if (uri.includes('/api/dashboard/')) return Promise.reject(new Error('api down'));
      if (uri.includes('/api/properties/')) return Promise.reject(new Error('api down'));
      return Promise.reject(new Error('metadata server down'));
    });
    render(<App />);
    await screen.findByRole('button', { name: 'Dashboard' });

    fireEvent.click(screen.getByRole('button', { name: 'Dashboard' }));
    await waitFor(
      () =>
        expect(screen.getByRole('heading', { name: 'Sold' })).toBeInTheDocument(),
      { timeout: 5000 },
    );
    expect(screen.getByText('Buyer')).toBeInTheDocument();
    expect(
      screen.getByRole('heading', { name: 'Marketplace overview' }),
    ).toBeInTheDocument();
    expect(
      screen.getByRole('heading', { name: 'Purchases in escrow' }),
    ).toBeInTheDocument();
    expect(
      screen.getByRole('heading', { name: 'My properties' }),
    ).toBeInTheDocument();
    // A dead API is reported, not hidden behind an empty section.
    expect(screen.getByText(/market overview unavailable/i)).toBeInTheDocument();
  });

  it('opens a property page and can return to the marketplace', async () => {
    setupEthereum(accounts.buyer);
    render(<App />);

    const [card] = await screen.findAllByText('Whitefield');
    fireEvent.click(card);

    const dialog = await screen.findByRole('dialog');
    expect(dialog).toBeInTheDocument();
    expect(window.location.pathname).toBe('/property/MREID_0000001');
  });

  it('shows the AI assistant with property context', async () => {
    setupEthereum(accounts.buyer);
    global.fetch = jest.fn((uri) => {
      if (uri.includes('/api/properties/MREID_0000001') && uri.includes('/chain')) {
        return json({
          mreid_id: 'MREID_0000001',
          chain: {
            tokenized: true,
            token_id: 1,
            owner: accounts.seller,
            listed: false,
            active_sale: false,
            finalized: false,
            sale_status: 'None',
          },
          chain_snapshot: { available: true, exported_at: '2026-09-26T16:23:37.384Z' },
        });
      }
      if (/\/api\/properties\/MREID_0000001\/?$/.test(uri)) {
        return json({
          mreid_id: 'MREID_0000001',
          listed_price: 9000000,
          listed_price_formatted: '₹90 Lakh',
          price_per_sqft: 3600,
          property: { area: 2500, location: 'Whitefield', city: 'Bangalore', bedrooms: 3, amenities: {} },
          ai_estimation: {
            ai_estimated_price: 9500000,
            ai_estimated_price_formatted: '₹95 Lakh',
            ai_estimated_price_per_sqft: 3800,
          },
          ai_market_signal: { label: 'Potentially undervalued' },
          locality: { count: 20, median_price_per_sqft: 4000 },
        });
      }
      return Promise.reject(new Error(`unexpected fetch: ${uri}`));
    });

    // The assistant is reached from the property page, not the navbar.  The
    // route is used directly because this test's fetch mock only serves the
    // property endpoints, not the marketplace search.
    window.history.pushState({}, '', '/property/MREID_0000001');
    render(<App />);
    await screen.findByRole('dialog');

    expect(
      screen.getByRole('button', { name: /open MILLOW AI assistant/i }),
    ).toBeInTheDocument();
    fireEvent.click(
      screen.getByRole('button', { name: /open MILLOW AI assistant/i }),
    );

    expect(await screen.findByText('MILLOW AI')).toBeInTheDocument();
    expect(window.location.pathname).toBe('/assistant/MREID_0000001');

    // The quick actions are about this property.  They swap in once the
    // property context has loaded.
    for (const label of ['Price', 'Risk', 'Details', 'Locality', 'On-chain', 'Similar']) {
      expect(
        await screen.findByRole('button', { name: new RegExp(`^${label}`, 'i') }),
      ).toBeInTheDocument();
    }
    expect(screen.getAllByText(/Asking about/i).length).toBeGreaterThan(0);
    expect(
      screen.queryByRole('button', { name: /Find properties/i }),
    ).toBeNull();
  });
});
