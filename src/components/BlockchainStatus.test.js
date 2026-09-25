import { render, screen } from "@testing-library/react";
import BlockchainStatus from "./BlockchainStatus";
import { getPropertyChainStatus, demoRoleOf } from "../lib/blockchain";

jest.mock("../lib/blockchain", () => ({
  getPropertyChainStatus: jest.fn(),
  demoRoleOf: jest.fn(() => null),
  ROLE_NAMES: {
    seller: "Seller",
    buyer: "Buyer",
    inspector: "Inspector",
    lender: "Lender",
    viewer: "Viewer",
  },
}));

const NFT = "0x1434acC7A5EC5FfaB6E0acEAa9dFC0233faE9A36";
const REGISTRY = "0x5FbDB2315678afecb367f032d93F642f64180aa3";
const OWNER = "0x70997970C51812dc3A010C7d01b50e0d17dc79c8";
const URI =
  "http://localhost:3000/metadata/millow/MREID_0000001.json";
const MINT_TX = "0x9c6a3a2cb6f1f4c9c967c0ad1d9e16fea90a1f710d2a6d7ed77e9b880b3578e7";

beforeEach(() => {
  jest.clearAllMocks();
  demoRoleOf.mockReturnValue(null);
});

describe("BlockchainStatus", () => {
  it("shows the loading state while checking the chain", () => {
    getPropertyChainStatus.mockReturnValue(new Promise(() => {}));
    render(<BlockchainStatus mreidId="MREID_0000001" />);
    expect(screen.getByText("Blockchain & NFT Status")).toBeInTheDocument();
    expect(
      screen.getByText("Checking the local blockchain…"),
    ).toBeInTheDocument();
  });

  it("shows a registered, on-offer token with on-chain data", async () => {
    getPropertyChainStatus.mockResolvedValue({
      available: true,
      chainId: 31337,
      chainName: "Millow Localhost (Hardhat Network)",
      supported: true,
      nftAddress: NFT,
      registryAddress: REGISTRY,
      mreidId: "MREID_0000001",
      registered: true,
      tokenId: 1,
      owner: OWNER,
      propertyId: "MREID_0000001",
      tokenURI: URI,
      onOffer: true,
      seller: OWNER,
      mintTxHash: MINT_TX,
      mintBlock: 17,
    });
    demoRoleOf.mockReturnValue("seller");

    render(<BlockchainStatus mreidId="MREID_0000001" />);

    expect(await screen.findByText("#1")).toBeInTheDocument();
    expect(screen.getByText("Blockchain & NFT Status")).toBeInTheDocument();
    expect(screen.getByText("Millow Localhost (Hardhat Network)")).toBeInTheDocument();
    expect(screen.getByText("31337")).toBeInTheDocument();
    expect(screen.getByText("#1")).toBeInTheDocument();
    expect(screen.getByText("MREID_0000001")).toBeInTheDocument();
    expect(screen.getByText("On offer")).toBeInTheDocument();
    expect(screen.getByText(URI)).toBeInTheDocument();
    expect(screen.getByText("0x9c6a3a2c…3578e7")).toBeInTheDocument();
    // The owner row names the connected market role next to the address.
    expect(screen.getByText(/(Seller)/)).toBeInTheDocument();
    expect(
      screen.getByText(/Digital property representation/),
    ).toBeInTheDocument();
  });

  it("shows a sold token as owned by the buyer with a Sold listing status", async () => {
    getPropertyChainStatus.mockResolvedValue({
      available: true,
      chainId: 31337,
      chainName: "Millow Localhost (Hardhat Network)",
      supported: true,
      nftAddress: NFT,
      registryAddress: REGISTRY,
      mreidId: "MREID_0000001",
      registered: true,
      tokenId: 1,
      owner: OWNER,
      propertyId: "MREID_0000001",
      tokenURI: URI,
      onOffer: false,
      seller: null,
      saleStatus: 4,
      finalized: true,
      mintTxHash: MINT_TX,
      mintBlock: 40,
    });
    demoRoleOf.mockReturnValue("buyer");

    render(<BlockchainStatus mreidId="MREID_0000001" />);

    expect(await screen.findByText("Sold")).toBeInTheDocument();
    expect(screen.getByText(/(Buyer)/)).toBeInTheDocument();
    expect(screen.queryByText("On offer")).toBeNull();
    expect(screen.queryByText("Not listed")).toBeNull();
  });

  it("shows the not-registered state when no token exists yet", async () => {
    getPropertyChainStatus.mockResolvedValue({
      available: true,
      chainId: 31337,
      chainName: "Millow Localhost (Hardhat Network)",
      supported: true,
      nftAddress: NFT,
      registryAddress: REGISTRY,
      mreidId: "MREID_0009999",
      registered: false,
    });

    render(<BlockchainStatus mreidId="MREID_0009999" />);

    expect(await screen.findByText("Not registered")).toBeInTheDocument();
    expect(screen.getByText("MREID_0009999")).toBeInTheDocument();
    expect(screen.getAllByText("—").length).toBeGreaterThan(0);
  });

  it("offers no register action and explains the unregistered state neutrally", async () => {
    getPropertyChainStatus.mockResolvedValue({
      available: true,
      chainId: 31337,
      chainName: "Millow Localhost (Hardhat Network)",
      supported: true,
      nftAddress: NFT,
      registryAddress: REGISTRY,
      mreidId: "MREID_0009999",
      registered: false,
    });
    render(<BlockchainStatus mreidId="MREID_0009999" />);

    expect(await screen.findByText("Not registered")).toBeInTheDocument();

    // The simplified workflow has no registrar and no register flow.
    expect(screen.getByText(/No NFT yet/)).toBeInTheDocument();
    expect(
      screen.getByText(/has no token on the prototype chain/),
    ).toBeInTheDocument();
    expect(
      screen.queryByRole("button", { name: /Register property NFT/ }),
    ).toBeNull();
    expect(screen.queryByText(/Registrar/)).toBeNull();
    expect(screen.queryByText(/MINTER_ROLE/)).toBeNull();
  });

  it("ignores legacy registration props — nothing register-ish ever renders", async () => {
    getPropertyChainStatus.mockResolvedValue({
      available: true,
      chainId: 31337,
      chainName: "Millow Localhost (Hardhat Network)",
      supported: true,
      nftAddress: NFT,
      registryAddress: REGISTRY,
      mreidId: "MREID_0009999",
      registered: false,
    });

    render(
      <BlockchainStatus
        mreidId="MREID_0009999"
        onRegister={jest.fn()}
        registerPending
        registerNotice="Registered MREID_0009999 as property NFT #8."
      />,
    );

    expect(await screen.findByText("Not registered")).toBeInTheDocument();
    expect(
      screen.queryByText("Registered MREID_0009999 as property NFT #8."),
    ).toBeNull();
    expect(
      screen.queryByText(/Only the authorized MILLOW registrar can register/),
    ).toBeNull();
  });

  it("shows the neutral unavailable state when the node is down", async () => {
    getPropertyChainStatus.mockResolvedValue({
      available: false,
      reason: "no-node",
      error: "Could not reach the local Hardhat node at http://127.0.0.1:8545.",
    });

    render(<BlockchainStatus mreidId="MREID_0000001" />);

    expect(
      await screen.findByText("Blockchain unavailable."),
    ).toBeInTheDocument();
    expect(
      screen.getByText(
        "Could not reach the local Hardhat node at http://127.0.0.1:8545.",
      ),
    ).toBeInTheDocument();
    expect(screen.getByText("Retry")).toBeInTheDocument();
    expect(
      screen.getByText(/Legal ownership remains subject/),
    ).toBeInTheDocument();
  });

  it("warns on an unsupported network", async () => {
    getPropertyChainStatus.mockResolvedValue({
      available: true,
      chainId: 1337,
      chainName: "Other",
      supported: false,
    });

    render(<BlockchainStatus mreidId="MREID_0000001" />);

    expect(
      await screen.findByText("Unsupported network."),
    ).toBeInTheDocument();
    expect(screen.getByText(/chain 1337/)).toBeInTheDocument();
  });

  it("never claims legal property ownership", async () => {
    getPropertyChainStatus.mockResolvedValue({
      available: true,
      chainId: 31337,
      chainName: "Millow Localhost (Hardhat Network)",
      supported: true,
      nftAddress: NFT,
      registryAddress: REGISTRY,
      mreidId: "MREID_0000001",
      registered: true,
      tokenId: 1,
      owner: OWNER,
      tokenURI: URI,
      onOffer: false,
      mintTxHash: MINT_TX,
    });

    render(<BlockchainStatus mreidId="MREID_0000001" />);

    await screen.findByText("Not listed");
    const body = document.body.textContent;
    expect(body).toContain("Digital property representation");
    expect(body).toContain(
      "Blockchain-based ownership record for the prototype",
    );
    expect(body).toContain("Technical proof-of-concept");
    expect(body).toContain(
      "Legal ownership remains subject to applicable Indian property and registration law.",
    );
    expect(body).not.toContain("NFT = legal ownership");
    expect(body).not.toContain("transfers title");
    expect(body).not.toContain("real money");
  });
});