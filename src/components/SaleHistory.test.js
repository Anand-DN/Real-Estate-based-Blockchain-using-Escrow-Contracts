import { render, screen } from "@testing-library/react";
import { ethers } from "ethers";
import SaleHistory from "./SaleHistory";
import { getSaleHistory } from "../lib/blockchain";

jest.mock("../lib/blockchain", () => ({
  getSaleHistory: jest.fn(),
}));

const base = {
  available: true,
  registered: true,
  supported: true,
  chainId: 31337,
  tokenId: 1,
};

beforeEach(() => {
  jest.clearAllMocks();
});

describe("SaleHistory", () => {
  it("shows an empty-state message when there are no events", async () => {
    getSaleHistory.mockResolvedValue({ ...base, rows: [] });

    render(<SaleHistory mreidId="MREID_0000001" />);

    expect(
      await screen.findByText(/has not been through an escrow sale yet/),
    ).toBeInTheDocument();
  });

  it("renders on-chain events with test-ETH amounts", async () => {
    getSaleHistory.mockResolvedValue({
      ...base,
      rows: [
        {
          key: "1:0:SaleListed",
          type: "SaleListed",
          txHash: "0xabc123",
          blockNumber: 10,
          amountWei: "100000000000000000000",
        },
        {
          key: "2:0:EarnestDeposited",
          type: "EarnestDeposited",
          txHash: "0xdef456",
          blockNumber: 12,
          amountWei: "30000000000000000000",
        },
        {
          key: "3:0:SaleFinalized",
          type: "SaleFinalized",
          txHash: "0x999111",
          blockNumber: 20,
          amountWei: "300000000000000000000",
        },
      ],
    });

    render(<SaleHistory mreidId="MREID_0000001" />);

    expect(await screen.findByText("Listed for sale")).toBeInTheDocument();
    expect(screen.getByText("Earnest deposit")).toBeInTheDocument();
    expect(screen.getByText("Finalized")).toBeInTheDocument();
    expect(screen.getByText("100 test ETH")).toBeInTheDocument();
    expect(screen.getByText("300 test ETH")).toBeInTheDocument();
    expect(screen.getByText("30 test ETH")).toBeInTheDocument();
    expect(screen.getByText("0xabc123")).toBeInTheDocument();
    expect(
      screen.getByText(/test ETH on the prototype chain and are not real money/),
    ).toBeInTheDocument();
  });

  it("renders inspector and lender role events with readable labels", async () => {
    getSaleHistory.mockResolvedValue({
      ...base,
      rows: [
        {
          key: "1:0:SaleListed",
          type: "SaleListed",
          txHash: "0xabc123",
          blockNumber: 10,
          amountWei: "100000000000000000000",
        },
        {
          key: "4:0:InspectionUpdated",
          type: "InspectionUpdated",
          txHash: "0xbbb222",
          blockNumber: 15,
          amountWei: null,
        },
        {
          key: "5:0:FinancingRequested",
          type: "FinancingRequested",
          txHash: "0xccc333",
          blockNumber: 16,
          amountWei: null,
        },
        {
          key: "6:0:LoanApproved",
          type: "LoanApproved",
          txHash: "0xddd444",
          blockNumber: 17,
          amountWei: null,
        },
        {
          key: "7:0:LoanFunded",
          type: "LoanFunded",
          txHash: "0xeee555",
          blockNumber: 18,
          amountWei: "240000000000000000000",
        },
        {
          key: "8:0:SellerApproved",
          type: "SellerApproved",
          txHash: "0xfff666",
          blockNumber: 19,
          amountWei: null,
        },
      ],
    });

    render(<SaleHistory mreidId="MREID_0000001" />);

    expect(
      await screen.findByText("Inspector verification updated"),
    ).toBeInTheDocument();
    expect(screen.getByText("Financing requested by buyer")).toBeInTheDocument();
    expect(screen.getByText("Loan approved")).toBeInTheDocument();
    expect(screen.getByText("Lender loan funded")).toBeInTheDocument();
    expect(screen.getByText("Seller approved")).toBeInTheDocument();
    expect(screen.getByText("240 test ETH")).toBeInTheDocument();
  });

  it("shows an unregistered empty state before tokenization", async () => {
    getSaleHistory.mockResolvedValue({
      available: true,
      registered: false,
      supported: true,
      chainId: 31337,
    });

    render(<SaleHistory mreidId="MREID_0009999" />);

    expect(
      await screen.findByText(/No blockchain activity yet/),
    ).toBeInTheDocument();
    expect(
      screen.getByText(/not tokenized on the prototype chain/),
    ).toBeInTheDocument();
  });

  it("labels the PropertyMinted event readably", async () => {
    getSaleHistory.mockResolvedValue({
      ...base,
      rows: [
        {
          key: "1:0:PropertyMinted",
          type: "PropertyMinted",
          txHash: "0xabc123",
          blockNumber: 10,
          amountWei: null,
        },
        {
          key: "2:0:SaleListed",
          type: "SaleListed",
          txHash: "0xdef456",
          blockNumber: 12,
          amountWei: "100000000000000000000",
        },
      ],
    });

    render(<SaleHistory mreidId="MREID_0000001" />);

    expect(
      await screen.findByText("NFT minted (registered)"),
    ).toBeInTheDocument();
    expect(screen.getByText("Listed for sale")).toBeInTheDocument();
  });

  it("renders a neutral unavailable box when the node is down", async () => {
    getSaleHistory.mockResolvedValue({
      available: false,
      reason: "no-node",
      error: "Could not reach the local Hardhat node.",
    });

    render(<SaleHistory mreidId="MREID_0000001" />);

    expect(
      await screen.findByText("Could not reach the local Hardhat node."),
    ).toBeInTheDocument();
  });
});