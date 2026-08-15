import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import App from './App';

jest.setTimeout(20000);

const accounts = {
  buyer: '0x70997970C51812dc3A010C7d01b50e0d17dc79C8',
  seller: '0x3C44CdDdB6a900fa2b585dd299e03d12FA4293BC',
  inspector: '0x90F79bf6EB2c4f870365E785982E1f101E93b906',
  lender: '0x15d34AAf54267DB7D7c367839AAf71A00a2C6A65',
};

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

jest.mock('ethers', () => {
  const actual = jest.requireActual('ethers');
  const { BigNumber } = actual;

  const contractValues = {
    totalSupply: () => Promise.resolve(BigNumber.from(6)),
    tokenURI: (id) => Promise.resolve(`http://localhost:3000/metadata/${id}.json`),
    inspector: () => Promise.resolve(accounts.inspector),
    lender: () => Promise.resolve(accounts.lender),
    isListed: () => Promise.resolve(true),
    ownerOf: () => Promise.resolve(accounts.seller),
    sellerOf: () => Promise.resolve(accounts.seller),
    buyer: () => Promise.resolve('0x0000000000000000000000000000000000000000'),
    inspectionPassed: () => Promise.resolve(false),
    approval: () => Promise.resolve(false),
    purchasePrice: () => Promise.resolve(BigNumber.from('20000000000000000000')),
    escrowAmount: () => Promise.resolve(BigNumber.from('10000000000000000000')),
    queryFilter: () => Promise.resolve([]),
    filters: { Transfer: () => ({}) },
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

describe('Dashboard repro', () => {
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
      const id = uri.match(/(\d+)\.json/)[1];
      return Promise.resolve({ ok: true, json: () => Promise.resolve(metadata(id)) });
    });
    localStorage.clear();
  });

  it('opens dashboard with buyer role and sections', async () => {
    setupEthereum(accounts.buyer);
    render(<App />);
    await screen.findAllByText('20 ETH');

    fireEvent.click(screen.getByRole('button', { name: 'Dashboard' }));
    await waitFor(() => expect(screen.getByText('My Dashboard')).toBeInTheDocument());
    await waitFor(() => expect(screen.getByText('Buyer')).toBeInTheDocument(), { timeout: 5000 });
    expect(screen.getByText('Purchase history')).toBeInTheDocument();
  });

  it('opens dashboard with seller role and properties for sale', async () => {
    setupEthereum(accounts.seller);
    render(<App />);
    await screen.findAllByText('20 ETH');

    fireEvent.click(screen.getByRole('button', { name: 'Dashboard' }));
    await waitFor(() => expect(screen.getByText('Seller')).toBeInTheDocument(), { timeout: 5000 });
    expect(screen.getByText('Properties for sale')).toBeInTheDocument();
    expect(screen.getByText('6')).toBeInTheDocument();
  });

  it('shows sections even when homes are empty', async () => {
    setupEthereum(accounts.buyer);
    global.fetch = jest.fn(() => Promise.reject(new Error('metadata server down')));
    render(<App />);
    await screen.findByRole('button', { name: 'Dashboard' });

    fireEvent.click(screen.getByRole('button', { name: 'Dashboard' }));
    await waitFor(() => expect(screen.getByText('Purchase history')).toBeInTheDocument(), { timeout: 5000 });
    expect(screen.getByText('Buyer')).toBeInTheDocument();
  });
});
