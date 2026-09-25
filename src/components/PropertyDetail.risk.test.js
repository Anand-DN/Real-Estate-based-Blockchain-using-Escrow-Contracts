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

const RISK = {
  mreid_id: 'MREID_0000001',
  analysis_subject: 'property_listing',
  anomaly_score: 37,
  score_scale: '0-100',
  indicators: [
    {
      id: 'listing_price_gap',
      title: 'Listing price per sqft vs comparables',
      dimension: 'pricing',
      status: 'medium',
      direction: 'above_typical',
      z: 3.1,
      explanation: 'Listed price vs median of 31 comparable listings.',
      severity_points: 2,
    },
    {
      id: 'amenity_disclosure',
      title: 'Amenity disclosure completeness',
      dimension: 'records',
      status: 'none',
      direction: 'n/a',
      z: null,
      explanation: '35 of 35 amenity attributes are disclosed.',
      severity_points: 0,
    },
  ],
  horizon_context: {
    city: 'Bengaluru',
    n_transactions: 2142,
    median_price_per_sqft: 24750,
    rapid_repeat_share: 0.04,
    synthetic_note:
      'City-level transaction statistics come from the Horizon dataset, which appears to be procedurally generated benchmark data.',
  },
  methodology: 'Indicators are computed from the property listing only, against comparable MREID listings.',
  disclaimer: 'This is a statistical anomaly analysis. It is not fraud detection, not a probability.',
};

const jsonResponse = (body, ok = true, status = 200) =>
  Promise.resolve({ ok, status, json: () => Promise.resolve(body) });

const installFetch = (riskImpl) => {
  global.fetch = jest.fn((url) => {
    if (String(url).includes('/market-context')) {
      return jsonResponse({
        mreid_id: 'MREID_0000001',
        property: { listed_price_per_sqft: 8982.04 },
        millow_ai: { ai_estimated_price_per_sqft: 4520 },
        market_context: {
          city: 'Bengaluru',
          composite_price_rs_sqft: 9567,
          note: 'NHB RESIDEX external benchmark.',
        },
      });
    }
    if (String(url).includes('/risk-analysis')) {
      return riskImpl();
    }
    return jsonResponse(DETAIL);
  });
};

describe('PropertyDetail Transaction Risk & Anomaly Analysis', () => {
  it('renders the risk section with score, indicators and explanations on success', async () => {
    installFetch(() => jsonResponse(RISK));

    render(<PropertyDetail mreidId="MREID_0000001" onClose={() => {}} />);

    expect(
      await screen.findByText('Transaction Risk & Anomaly Analysis'),
    ).toBeInTheDocument();

    expect(screen.getByText('37')).toBeInTheDocument();
    expect(
      screen.getByText(/How unusual this listing is vs comparable/),
    ).toBeInTheDocument();

    expect(
      screen.getByText('Listing price per sqft vs comparables'),
    ).toBeInTheDocument();
    expect(
      screen.getByText(/Listed price vs median of 31 comparable listings/),
    ).toBeInTheDocument();
    expect(
      screen.getByText('Amenity disclosure completeness'),
    ).toBeInTheDocument();

    expect(screen.getByText('medium')).toBeInTheDocument();
    expect(screen.getByText('none')).toBeInTheDocument();

    expect(screen.getByText(/robust z: 3.1/)).toBeInTheDocument();
    expect(screen.getByText(/above typical/)).toBeInTheDocument();

    expect(screen.getByText(/It is not fraud detection/)).toBeInTheDocument();
    expect(
      screen.getByText(/About this analysis/),
    ).toBeInTheDocument();
  });

  it('renders Horizon city-level context when available', async () => {
    installFetch(() => jsonResponse(RISK));

    render(<PropertyDetail mreidId="MREID_0000001" onClose={() => {}} />);

    await screen.findByText('Transaction Risk & Anomaly Analysis');
    expect(
      screen.getByText('City-level transaction context (Bengaluru)'),
    ).toBeInTheDocument();
    expect(screen.getByText(/2,142/)).toBeInTheDocument();
    expect(screen.getByText(/4% rapid\n* repeat sales/)).toBeInTheDocument();
    expect(
      screen.getByText(/procedurally generated benchmark data/),
    ).toBeInTheDocument();
  });

  it('shows loading state while the risk request is pending', async () => {
    installFetch(() => new Promise(() => {}));

    render(<PropertyDetail mreidId="MREID_0000001" onClose={() => {}} />);

    await screen.findByText(/Loading anomaly analysis/);
    expect(screen.getByText('₹3.00 Crore')).toBeInTheDocument();
  });

  it('shows an error state and keeps the main property detail visible', async () => {
    installFetch(() =>
      Promise.resolve({
        ok: false,
        status: 500,
        json: () => Promise.resolve({ detail: 'risk analysis exploded' }),
      }),
    );

    render(<PropertyDetail mreidId="MREID_0000001" onClose={() => {}} />);

    await screen.findByText('Could not load the anomaly analysis.');
    expect(screen.getByText(/risk analysis exploded/)).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Retry' })).toBeInTheDocument();
    expect(screen.getByText('₹3.00 Crore')).toBeInTheDocument();
  });
});