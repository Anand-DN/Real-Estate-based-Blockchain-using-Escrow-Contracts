import { render, screen, waitFor } from "@testing-library/react";
import { ethers } from "ethers";
import PropertyDetail from "./PropertyDetail";
import { getSaleStatus, getPropertyChainStatus } from "../lib/blockchain";

jest.mock("../lib/blockchain", () => ({
  getSaleStatus: jest.fn(),
  getPropertyChainStatus: jest.fn(),
  getSaleHistory: jest.fn(() => Promise.resolve({ available: true, events: [] })),
  demoRoleOf: jest.fn(() => null),
  ROLE_NAMES: {
    seller: "Seller",
    buyer: "Buyer",
    inspector: "Inspector",
    lender: "Lender",
    viewer: "Viewer",
  },
}));

jest.mock("../lib/millowApi", () => ({
  propertyById: jest.fn(),
  marketContext: jest.fn(),
  recommendations: jest.fn(),
  riskAnalysis: jest.fn(),
  cities: jest.fn(() => Promise.resolve([])),
  locations: jest.fn(() => Promise.resolve([])),
  searchProperties: jest.fn(() =>
    Promise.resolve({ total: 0, page: 1, total_pages: 1, results: [] }),
  ),
}));

import { propertyById, marketContext, riskAnalysis, recommendations } from "../lib/millowApi";

const ETH = (n) =>
  ethers.BigNumber.from(n).mul(ethers.utils.parseEther("1"));

const ZERO = "0x0000000000000000000000000000000000000000";

const saleState = (mreidId) => ({
  available: true,
  registered: true,
  supported: true,
  chainId: 31337,
  tokenId: mreidId === "MREID_0000002" ? 2 : 1,
  onOffer: true,
  status: 2,
  statusName: "Under contract",
  seller: "0x70997970C51812dc3A010C7d01b50e0d17dc79C8",
  buyer: ZERO,
  priceWei: mreidId === "MREID_0000002" ? ETH(400) : ETH(300),
  earnestWei: ETH(30),
  buyerFundedWei: ETH(0),
  lenderFundedWei: ETH(0),
  sellerApproved: false,
  buyerApproved: false,
  inspectionPassed: false,
  inspectionRequired: mreidId === "MREID_0000002",
  lenderRequired: mreidId === "MREID_0000002",
  financing: {
    lender: ZERO,
    requested: false,
    approved: false,
    rejected: false,
    downPaymentPctBps: 0,
    interestRateBps: 0,
    loanTenureMonths: 0,
    downPaymentWei: ETH(0),
    loanAmountWei: ETH(0),
  },
  accountRole: "viewer",
});

const detailFor = (mreidId) => ({
  mreid_id: mreidId,
  property: {
    city: mreidId === "MREID_0000002" ? "Bangalore" : "Bangalore",
    location: mreidId === "MREID_0000002" ? "Whitefield" : "Koramangala",
    area: 2000,
    bedrooms: 3,
    resale: 1,
    amenities: { GYM: "No" },
  },
  listed_price_formatted: mreidId === "MREID_0000002" ? "₹4.00 Crore" : "₹3.00 Crore",
  price_per_sqft: 10000,
  ai_estimation: {
    ai_estimated_price_formatted: "₹2.00 Crore",
    ai_estimated_price_per_sqft: 10000,
    model: "MILLOW V5 Log-Price XGBoost",
    note: "AI-assisted estimate for research purposes only.",
  },
  disclaimer: "Disclaimer text.",
});

describe("PropertyDetail navigation stays per-property (no stale cross-property state)", () => {
  beforeEach(() => {
    jest.clearAllMocks();
    getPropertyChainStatus.mockResolvedValue({
      available: true,
      registered: true,
      supported: true,
      chainId: 31337,
      owner: "0x70997970C51812dc3A010C7d01b50e0d17dc79C8",
      tokenId: 1,
      onOffer: true,
    });
    propertyById.mockImplementation((mreidId) =>
      Promise.resolve(detailFor(mreidId)),
    );
    marketContext.mockRejectedValue(new Error("no market"));
    riskAnalysis.mockRejectedValue(new Error("no risk"));
    recommendations.mockResolvedValue({ items: [] });
  });

  it("shows MREID_0000001 -> MREID_0000002 -> MREID_0000001 with correct per-property data", async () => {
    getSaleStatus.mockImplementation((mreidId) =>
      Promise.resolve(saleState(mreidId)),
    );

    const { rerender } = render(
      <PropertyDetail mreidId="MREID_0000001" onClose={() => {}} />,
    );

    expect(await screen.findByText("Koramangala")).toBeInTheDocument();
    expect(await screen.findByText("300 test ETH")).toBeInTheDocument();
    expect(screen.queryByText("400 test ETH")).toBeNull();

    rerender(<PropertyDetail mreidId="MREID_0000002" onClose={() => {}} />);

    expect(await screen.findByText("Whitefield")).toBeInTheDocument();
    // Price shows in the txn summary stat and the purchase-plan row for a
    // lender-required sale (Model B), so use getAllByText.
    expect(
      (await screen.findAllByText("400 test ETH")).length,
    ).toBeGreaterThan(0);
    expect(screen.queryByText("300 test ETH")).toBeNull();
    expect(screen.queryByText(/Model A — no inspection/)).toBeNull();

    rerender(<PropertyDetail mreidId="MREID_0000001" onClose={() => {}} />);

    expect(await screen.findByText("Koramangala")).toBeInTheDocument();
    expect(await screen.findByText("300 test ETH")).toBeInTheDocument();
    expect(screen.queryByText("400 test ETH")).toBeNull();
  });

  it("does not flash the previous property's on-chain state while the next fetch is pending", async () => {
    let resolveNext;
    const next = new Promise((resolve) => {
      resolveNext = resolve;
    });
    getSaleStatus.mockImplementation((mreidId) =>
      mreidId === "MREID_0000002" ? next : Promise.resolve(saleState(mreidId)),
    );

    const { rerender } = render(
      <PropertyDetail mreidId="MREID_0000001" onClose={() => {}} />,
    );
    expect(await screen.findByText("300 test ETH")).toBeInTheDocument();

    rerender(<PropertyDetail mreidId="MREID_0000002" onClose={() => {}} />);

    await waitFor(() => {
      expect(screen.queryByText("300 test ETH")).toBeNull();
    });
    expect(screen.queryByText("400 test ETH")).toBeNull();
    expect(screen.queryByText("Purchase Progress")).toBeNull();

    resolveNext(saleState("MREID_0000002"));
    expect(
      (await screen.findAllByText("400 test ETH")).length,
    ).toBeGreaterThan(0);
    expect(screen.queryByText("300 test ETH")).toBeNull();
  });
});