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

const REC_CARD = {
  mreid_id: 'MREID_0000224',
  bedrooms: 4,
  area: 3300,
  location: 'JP Nagar Phase 1',
  city: 'Bangalore',
  price: 29600000,
  price_formatted: '₹2.96 Crore',
  price_per_sqft: 8970.1,
  amenities: { LIFT: 'Yes', PARK: 'No' },
  ai_estimated_price_formatted: '₹1.55 Crore',
  ai_estimated_price_per_sqft: 4700,
};

const RECS = {
  mode: 'similar_to_property',
  count: 2,
  items: [
    {
      mreid_id: 'MREID_0000224',
      recommendation_score: 100,
      reasons: [
        'Same locality: JP Nagar Phase 1',
        'Similar 4 bedrooms',
        '14 matching amenities',
      ],
      property: { ...REC_CARD },
    },
    {
      mreid_id: 'MREID_0004390',
      recommendation_score: 90,
      reasons: ['Compared against Bangalore listings', 'Similar 4 bedrooms'],
      property: {
        ...REC_CARD,
        mreid_id: 'MREID_0004390',
        location: 'Indiranagar',
        price_formatted: '₹2.60 Crore',
      },
    },
  ],
  methodology: 'Candidates are scored on locality exact-match...',
  disclaimer: 'Recommendations are similarity-based matches, not investment advice.',
};

const jsonResponse = (body, ok = true, status = 200) =>
  Promise.resolve({ ok, status, json: () => Promise.resolve(body) });

const installFetch = (recsImpl) => {
  global.fetch = jest.fn((url) => {
    if (String(url).includes('/recommendations')) {
      return recsImpl();
    }
    if (String(url).includes('/market-context')) {
      return jsonResponse({
        property: { listed_price_per_sqft: 8982.04 },
        millow_ai: { ai_estimated_price_per_sqft: 4520 },
        market_context: { city: 'Bengaluru', composite_price_rs_sqft: 9567 },
      });
    }
    return jsonResponse(DETAIL);
  });
};

describe('PropertyDetail Similar Properties', () => {
  it('renders the Similar Properties section with score, reasons and cards on success', async () => {
    installFetch(() => jsonResponse(RECS));

    render(
      <PropertyDetail mreidId="MREID_0000001" onClose={() => {}} />,
    );

    expect(await screen.findByText('Similar Properties')).toBeInTheDocument();

    expect(screen.getByText('100')).toBeInTheDocument();
    expect(screen.getByText('90')).toBeInTheDocument();
    expect(screen.getByText('JP Nagar Phase 1')).toBeInTheDocument();
    expect(screen.getByText('Indiranagar')).toBeInTheDocument();

    expect(
      screen.getByText('Same locality: JP Nagar Phase 1'),
    ).toBeInTheDocument();
    expect(screen.getByText('14 matching amenities')).toBeInTheDocument();
    expect(screen.getAllByText('Similar 4 bedrooms').length).toBeGreaterThan(0);

    expect(screen.getByText('₹2.96 Crore')).toBeInTheDocument();
  });

  it('renders an empty state when no similar properties exist', async () => {
    installFetch(() => jsonResponse({ mode: 'similar_to_property', count: 0, items: [] }));

    render(<PropertyDetail mreidId="MREID_0000001" onClose={() => {}} />);

    await screen.findByText('No similar properties found.');
    expect(screen.getByText('Similar Properties')).toBeInTheDocument();
  });

  it('shows loading state while the recommendations request is pending', async () => {
    installFetch(() => new Promise(() => {}));

    render(<PropertyDetail mreidId="MREID_0000001" onClose={() => {}} />);

    await screen.findByText(/Finding similar properties/);
    expect(screen.getByText('₹3.00 Crore')).toBeInTheDocument();
  });

  it('shows an error state and keeps the main property detail visible', async () => {
    installFetch(() =>
      Promise.resolve({
        ok: false,
        status: 500,
        json: () => Promise.resolve({ detail: 'recommendations exploded' }),
      }),
    );

    render(<PropertyDetail mreidId="MREID_0000001" onClose={() => {}} />);

    await screen.findByText('Could not load similar properties.');
    expect(screen.getByText(/recommendations exploded/)).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Retry' })).toBeInTheDocument();
    expect(screen.getByText('₹3.00 Crore')).toBeInTheDocument();
  });
});