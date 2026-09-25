import { render, screen } from '@testing-library/react';
import PropertyDetail from './PropertyDetail';

const DETAIL = {
  mreid_id: 'MREID_0000001',
  property: {
    city: 'Bangalore',
    location: 'Koramangala',
    area: 3340,
    bedrooms: 2,
    resale: 1,
    amenities: { GYM: 'No', PARKING: 'Yes' },
  },
  listed_price_formatted: '₹3.00 Crore',
  price_per_sqft: 8982.04,
  ai_estimation: {
    ai_estimated_price_formatted: '₹1.51 Crore',
    ai_estimated_price_per_sqft: 4520,
    model: 'MILLOW V5 Log-Price XGBoost',
    note: 'AI-assisted estimate for research purposes only.',
  },
  disclaimer: 'Disclaimer text.',
};

const MARKET = {
  mreid_id: 'MREID_0000001',
  city: 'Bangalore',
  nhb_city: 'Bengaluru',
  property: {
    listed_price: 30000000,
    area_sqft: 3340,
    listed_price_per_sqft: 8982.04,
  },
  millow_ai: {
    ai_estimated_price_per_sqft: 4520,
    model: 'MILLOW V5 Log-Price XGBoost',
  },
  market_context: {
    city: 'Bengaluru',
    quarter: 'Jun 2026',
    composite_price_rs_sqft: 9567,
    price_le_60sqm_rs_sqft: 7308,
    price_60_110sqm_rs_sqft: 8670,
    price_gt_110sqm_rs_sqft: 9567,
    hpi_assessment: 109.9,
    applicable_size_band: {
      label: '> 110 sqm',
      price_rs_sqft: 9567,
      property_area_sqft: 3340,
    },
    source: 'NHB RESIDEX',
    source_url: 'https://residex.nhbonline.org.in/',
    series: 'FY 2024-25 base year/current series',
    access_period: 'Jun 2025 to Jun 2026',
    note: 'NHB RESIDEX is an external, government-published city-level market benchmark.',
  },
  disclaimer: 'Disclaimer text.',
};

const jsonResponse = (body, ok = true, status = 200) =>
  Promise.resolve({ ok, status, json: () => Promise.resolve(body) });

const installFetch = (marketImpl) => {
  global.fetch = jest.fn((url) => {
    if (String(url).includes('/market-context')) {
      return marketImpl();
    }
    return jsonResponse(DETAIL);
  });
};

describe('PropertyDetail NHB Market Context', () => {
  it('renders the NHB Market Context section alongside existing content on success', async () => {
    installFetch(() => jsonResponse(MARKET));

    render(<PropertyDetail mreidId="MREID_0000001" onClose={() => {}} />);

    expect(
      await screen.findByText('NHB Market Context'),
    ).toBeInTheDocument();

    expect(screen.getByText('Bengaluru (NHB RESIDEX)')).toBeInTheDocument();
    expect(screen.getByText('Jun 2026')).toBeInTheDocument();
    expect(screen.getAllByText('₹9,567/sqft').length).toBeGreaterThan(0);
    expect(screen.getByText('109.9')).toBeInTheDocument();
    expect(screen.getAllByText(/NHB RESIDEX/).length).toBeGreaterThan(0);
    expect(screen.getAllByText(/FY 2024-25/).length).toBeGreaterThan(0);
    expect(screen.getByText(/city-level index, not this property/)).toBeInTheDocument();

    expect(screen.getByText('Listed price')).toBeInTheDocument();
    expect(screen.getByText('₹3.00 Crore')).toBeInTheDocument();
    expect(screen.getAllByText(/MILLOW AI estimate/).length).toBeGreaterThan(0);
    expect(screen.getByText('Amenities')).toBeInTheDocument();
    expect(screen.getByText('About this estimate')).toBeInTheDocument();
  });

  it('shows loading state while the market-context request is pending', async () => {
    installFetch(() => new Promise(() => {}));

    render(<PropertyDetail mreidId="MREID_0000001" onClose={() => {}} />);

    await screen.findByText(/Loading city-level NHB RESIDEX benchmark/);
    expect(screen.getByText('₹3.00 Crore')).toBeInTheDocument();
  });

  it('shows an error state and keeps the main property detail visible', async () => {
    installFetch(() =>
      Promise.resolve({
        ok: false,
        status: 500,
        json: () => Promise.resolve({ detail: 'market context exploded' }),
      }),
    );

    render(<PropertyDetail mreidId="MREID_0000001" onClose={() => {}} />);

    await screen.findByText('Could not load NHB market context.');
    expect(screen.getByText(/market context exploded/)).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Retry' })).toBeInTheDocument();
    expect(screen.getByText('₹3.00 Crore')).toBeInTheDocument();
  });
});