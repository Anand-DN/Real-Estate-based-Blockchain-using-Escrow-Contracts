import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { ethers } from 'ethers';
import App from './App';

jest.setTimeout(60000);

describe('Dashboard live integration', () => {
  let node;

  beforeAll(() => {
    node = new ethers.providers.JsonRpcProvider('http://localhost:8545');
    window.ethereum = {
      request: (args) => node.send(args.method, args.params || []),
      on: () => {},
      removeListener: () => {},
    };
  });

  it('loads real homes and renders a dashboard for the first hardhat account', async () => {
    const accounts = await node.send('eth_accounts', []);
    const account = ethers.utils.getAddress(accounts[0]);

    render(<App />);

    await waitFor(() => expect(screen.getAllByText(/ETH/).length).toBeGreaterThan(0), { timeout: 30000 });
    const ethBadges = screen.getAllByText(/ETH/);
    expect(ethBadges.length).toBeGreaterThanOrEqual(6);

    fireEvent.click(screen.getByRole('button', { name: 'Dashboard' }));

    await waitFor(() => expect(screen.queryByText('Loading your dashboard...')).not.toBeInTheDocument(), { timeout: 30000 });
    expect(screen.getByText('My Dashboard')).toBeInTheDocument();
    expect(screen.getAllByText(account.slice(0, 6) + '...' + account.slice(38, 42)).length).toBeGreaterThanOrEqual(1);
  });

  it('shows a useful role section instead of an empty hero', async () => {
    render(<App />);
    await waitFor(() => expect(screen.getAllByText(/ETH/).length).toBeGreaterThanOrEqual(6), { timeout: 30000 });

    fireEvent.click(screen.getByRole('button', { name: 'Dashboard' }));
    await waitFor(() => expect(screen.queryByText('Loading your dashboard...')).not.toBeInTheDocument(), { timeout: 30000 });

    const sections = screen.getAllByText(/You don't own any|No properties|You haven't listed|Properties for sale|Owned|Purchase history|Buying \(in escrow\)|Sold|No loans|No inspections/);
    expect(sections.length).toBeGreaterThan(0);
  });
});
