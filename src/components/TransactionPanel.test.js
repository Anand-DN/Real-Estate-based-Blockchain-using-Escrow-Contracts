import { render, screen, act, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { ethers } from "ethers";
import TransactionPanel from "./TransactionPanel";
import {
  getSaleStatus,
  getPropertyChainStatus,
  getSaleHistory,
  demoRoleOf,
} from "../lib/blockchain";
import { connectWallet, getSigner, subscribeWallet } from "../lib/wallet";

jest.mock("../lib/blockchain", () => ({
  getSaleStatus: jest.fn(),
  getPropertyChainStatus: jest.fn(),
  getSaleHistory: jest.fn(),
  demoRoleOf: jest.fn(() => null),
  ROLE_NAMES: {
    seller: "Seller",
    buyer: "Buyer",
    inspector: "Inspector",
    lender: "Lender",
    viewer: "Viewer",
  },
}));

jest.mock("../lib/wallet", () => ({
  connectWallet: jest.fn(),
  getSigner: jest.fn(() => Promise.resolve(null)),
  hasWallet: jest.fn(() => false),
  subscribeWallet: jest.fn(({ onAccount, onNetwork }) => {
    mockSubscribeHandlers = { onAccount, onNetwork };
    return () => {};
  }),
}));

let mockSubscribeHandlers = {};

const ETH = (n) =>
  ethers.BigNumber.from(n).mul(ethers.utils.parseEther("1"));

const ZERO = "0x0000000000000000000000000000000000000000";
// Canonical demo identities (see src/config.json demoRoles).
const SELLER = "0x70997970C51812dc3A010C7d01b50e0d17dc79C8";
const BUYER = "0xf39Fd6e51aad88F6F4ce6aB8827279cffFb92266";
const INSPECTOR = "0x3C44CdDdB6a900fa2b585dd299e03d12FA4293BC";
const LENDER = "0x90F79bf6EB2c4f870365E785982E1f101E93b906";
const REGISTRAR = "0xBcd4042DE499D14e55001CcbB24a551F3b954096";

const noFinancing = (overrides = {}) => ({
  lender: ZERO,
  requested: false,
  approved: false,
  rejected: false,
  downPaymentPctBps: 0,
  interestRateBps: 0,
  loanTenureMonths: 0,
  downPaymentWei: ETH(0),
  loanAmountWei: ETH(0),
  ...overrides,
});

const listedStatus = (overrides = {}) => ({
  available: true,
  registered: true,
  supported: true,
  chainId: 31337,
  tokenId: 1,
  onOffer: true,
  status: 1,
  statusName: "Listed for sale",
  seller: SELLER,
  buyer: ZERO,
  ownerOf: SELLER,
  priceWei: ETH(300),
  earnestWei: ETH(30),
  buyerFundedWei: ETH(0),
  lenderFundedWei: ETH(0),
  fundedWei: ETH(0),
  sellerApproved: false,
  buyerApproved: false,
  inspectionPassed: false,
  inspectionRequired: false,
  lenderRequired: false,
  financing: noFinancing(),
  ...overrides,
});

const chainStatus = (overrides = {}) => ({
  available: true,
  registered: true,
  supported: true,
  chainId: 31337,
  owner: SELLER,
  tokenId: 1,
  onOffer: true,
  ...overrides,
});

beforeEach(() => {
  jest.clearAllMocks();
  getSaleStatus.mockResolvedValue(listedStatus());
  getPropertyChainStatus.mockResolvedValue(chainStatus());
  getSaleHistory.mockResolvedValue({
    available: true,
    registered: true,
    chainId: 31337,
    rows: [],
  });
  demoRoleOf.mockReturnValue(null);
  subscribeWallet.mockImplementation(({ onAccount, onNetwork }) => {
    mockSubscribeHandlers = { onAccount, onNetwork };
    return () => {};
  });
});

const connect = async (address) => {
  await userEvent.click(
    await screen.findByRole("button", { name: /Connect wallet/ }),
  );
  if (address) {
    await screen.findByText(address);
  }
};

describe("TransactionPanel", () => {
  it("shows a neutral unavailable state when the node is down", async () => {
    getSaleStatus.mockResolvedValue({
      available: false,
      reason: "no-node",
      error: "Could not reach the local Hardhat node.",
    });

    render(<TransactionPanel mreidId="MREID_0000001" />);

    expect(
      await screen.findByText("Could not reach the local Hardhat node."),
    ).toBeInTheDocument();
  });

  it("warns when the connected chain is unsupported", async () => {
    getSaleStatus.mockResolvedValue({
      available: true,
      supported: false,
      chainId: 1337,
    });

    render(<TransactionPanel mreidId="MREID_0000001" />);

    expect(
      await screen.findByText(/chain \(1337\) is not supported/),
    ).toBeInTheDocument();
  });

  it("shows an informative unregistered workspace with no registration action", async () => {
    getSaleStatus.mockResolvedValue({
      available: true,
      registered: false,
      supported: true,
      chainId: 31337,
    });

    render(<TransactionPanel mreidId="MREID_0009999" />);

    expect(
      await screen.findByText(/MREID_0009999 is not yet tokenized/),
    ).toBeInTheDocument();
    expect(
      screen.getByText(/No transaction workflow/),
    ).toBeInTheDocument();
    expect(
      screen.queryByRole("button", { name: /Register property NFT/ }),
    ).toBeNull();
    // The simplified workflow has no Registrar and no register flow at all.
    expect(screen.queryByText(/Registrar/)).toBeNull();
    expect(screen.queryByText(/MINTER_ROLE/)).toBeNull();
    expect(
      screen.queryByText(/Switch to the Registrar account/),
    ).toBeNull();
  });

  it("shows the buyer a purchase plan where down payment + loan always equals the price", async () => {
    connectWallet.mockResolvedValue({ ok: true, account: BUYER });
    getSaleStatus.mockResolvedValue(
      listedStatus({
        status: 2,
        statusName: "Under contract",
        buyer: BUYER,
        buyerFundedWei: ETH(30),
        lenderRequired: true,
        inspectionRequired: true,
        inspectionPassed: true,
      }),
    );

    render(<TransactionPanel mreidId="MREID_0000001" />);
    await connect();

    expect(
      await screen.findByText(/Purchase plan \(down payment \+ loan = price\)/),
    ).toBeInTheDocument();
    // Default down payment = 20% of 300 = 60; requested loan is exactly the
    // remainder = 240.  The two can never sum to more or less than the price.
    expect(screen.getAllByText("60 test ETH").length).toBeGreaterThan(0);
    expect(screen.getByText("240 test ETH")).toBeInTheDocument();
    expect(
      screen.getByText(
        /Down payment \+ requested loan = property price: 60 test ETH \+ 240 test ETH = 300 test ETH/,
      ),
    ).toBeInTheDocument();
    expect(
      screen.getByText(/it can never exceed or fall short of the asking price/),
    ).toBeInTheDocument();
  });

  it("shows the compact 6-stage purchase progress with the next pending stage", async () => {
    connectWallet.mockResolvedValue({ ok: true, account: BUYER });
    getSaleStatus.mockResolvedValue(
      listedStatus({
        status: 2,
        statusName: "Under contract",
        buyer: BUYER,
        buyerFundedWei: ETH(30),
        inspectionRequired: true,
        lenderRequired: true,
      }),
    );

    render(<TransactionPanel mreidId="MREID_0000001" />);
    await connect();

    // Listed and committed, but the buyer has not chosen a payment plan yet.
    expect(
      await screen.findByText(/Next step: Buyer chooses payment plan/),
    ).toBeInTheDocument();
    expect(screen.getByText("Purchase Progress")).toBeInTheDocument();
    expect(screen.getByText("Seller lists property")).toBeInTheDocument();
    expect(screen.getByText("Buyer chooses payment plan")).toBeInTheDocument();
    expect(screen.getByText("Property inspection")).toBeInTheDocument();
    expect(screen.getByText("Loan approval")).toBeInTheDocument();
    expect(screen.getByText("Seller approval")).toBeInTheDocument();
    expect(screen.getByText("Ownership transfer")).toBeInTheDocument();
  });

  it("shows a compact finalized view with the ownership transfer complete", async () => {
    connectWallet.mockResolvedValue({ ok: true, account: BUYER });
    getSaleStatus.mockResolvedValue(
      listedStatus({
        status: 4,
        statusName: "Finalized",
        buyer: BUYER,
        ownerOf: BUYER,
        seller: SELLER,
        buyerFundedWei: ETH(60),
        lenderFundedWei: ETH(240),
        inspectionRequired: true,
        inspectionPassed: true,
        lenderRequired: true,
        financing: noFinancing({
          requested: true,
          approved: true,
          downPaymentPctBps: 2000,
          downPaymentWei: ETH(60),
          loanAmountWei: ETH(240),
        }),
        sellerApproved: true,
        buyerApproved: true,
      }),
    );

    render(<TransactionPanel mreidId="MREID_0000001" />);
    await connect();

    expect(
      await screen.findByText(
        /Sale finalized - the NFT was delivered to the buyer/,
      ),
    ).toBeInTheDocument();
    expect(screen.getByText("Transaction Finalized")).toBeInTheDocument();
    expect(screen.getByText("SOLD")).toBeInTheDocument();
    expect(screen.getByText("Buyer (NFT delivered)")).toBeInTheDocument();
    expect(
      screen.getByText(/₹3\.00 Crore \(300 test ETH\)/),
    ).toBeInTheDocument();
    expect(
      screen.getByText(/₹60 lakh \(60 test ETH\)/),
    ).toBeInTheDocument();
    expect(
      screen.getByText(/₹2\.40 Crore \(240 test ETH\)/),
    ).toBeInTheDocument();
  });

  it("keeps the role cards honest when a registered token is not listed", async () => {
    connectWallet.mockResolvedValue({ ok: true, account: SELLER });
    getSaleStatus.mockResolvedValue(
      listedStatus({
        onOffer: false,
        status: 0,
        statusName: "Not listed",
        accountRole: "seller",
      }),
    );
    getPropertyChainStatus.mockResolvedValue(
      chainStatus({ owner: SELLER, onOffer: false }),
    );
    demoRoleOf.mockImplementation((acc) =>
      String(acc).toLowerCase() === SELLER.toLowerCase() ? "seller" : null,
    );

    render(<TransactionPanel mreidId="MREID_0000001" />);
    await connect();

    // There is no Registrar / transfer card any more.
    expect(
      screen.queryByRole("button", { name: /Transfer NFT to the MILLOW seller/ }),
    ).toBeNull();
    expect(screen.queryByText("Registrar")).toBeNull();
    // The seller owns the token, so the segregated listing form is available.
    expect(
      await screen.findByRole("button", { name: /List Property for Sale/ }),
    ).toBeInTheDocument();
  });

  it("displays the INR-derived demo-test-ETH price and earnest with the demo-rate notice", async () => {
    render(<TransactionPanel mreidId="MREID_0000001" />);

    expect(await screen.findByText("300 test ETH")).toBeInTheDocument();
    expect(screen.getByText("30 test ETH")).toBeInTheDocument();
    expect(screen.getByText(/₹1,00,000 = 1 test ETH/)).toBeInTheDocument();
    expect(
      screen.getByText(/test ETH on Hardhat chain 31337/),
    ).toBeInTheDocument();
    expect(screen.getByText(/No real money/)).toBeInTheDocument();
    expect(
      screen.getByText(/legal ownership remains subject/),
    ).toBeInTheDocument();
  });

  it("reveals the listing form to the owner after connecting the wallet", async () => {
    connectWallet.mockResolvedValue({ ok: true, account: SELLER });
    getSaleStatus.mockResolvedValue(
      listedStatus({ onOffer: false, status: 0, statusName: "Not listed" }),
    );

    render(<TransactionPanel mreidId="MREID_0000001" />);
    await connect();

    expect(
      await screen.findByRole("button", { name: /List Property for Sale/ }),
    ).toBeInTheDocument();
  });

  it("shows the seller NO editable price input when listing (A)", async () => {
    connectWallet.mockResolvedValue({ ok: true, account: SELLER });
    getSaleStatus.mockResolvedValue(
      listedStatus({ onOffer: false, status: 0, statusName: "Not listed" }),
    );

    render(
      <TransactionPanel mreidId="MREID_0000001" listedPriceInr={30000000} />,
    );
    await connect();

    expect(
      await screen.findByRole("button", { name: /List Property for Sale/ }),
    ).toBeInTheDocument();
    expect(
      screen.queryByLabelText(/INR asking price/),
    ).toBeNull();
    expect(screen.queryByText(/INR asking price/)).toBeNull();
    expect(
      screen.queryByLabelText(/INR earnest/),
    ).toBeNull();
    // The checkboxes toggle workflow requirements, not prices — they stay.
    expect(
      screen.getByRole("checkbox", { name: /Require inspector verification/ }),
    ).toBeInTheDocument();
    expect(
      screen.getByRole("checkbox", { name: /Require lender financing/ }),
    ).toBeInTheDocument();
  });

  it("lets the seller list using the existing dataset price and shows it read-only (B, C)", async () => {
    connectWallet.mockResolvedValue({ ok: true, account: SELLER });
    getSigner.mockResolvedValue({});
    getSaleStatus.mockResolvedValue(
      listedStatus({
        onOffer: false,
        status: 0,
        statusName: "Not listed",
      }),
    );

    const listForSaleWithRequirements = jest.fn().mockResolvedValue(null);
    const approve = jest
      .fn()
      .mockResolvedValue({ wait: jest.fn().mockResolvedValue(undefined) });
    const contractSpy = jest.spyOn(ethers, "Contract");
    contractSpy.mockImplementation((addr, abi, providerOrSigner) => ({
      getApproved: jest.fn().mockResolvedValue(ZERO),
      approve,
      listForSaleWithRequirements,
      listForSale: jest.fn().mockResolvedValue(null),
    }));

    render(
      <TransactionPanel mreidId="MREID_0000001" listedPriceInr={30000000} />,
    );
    await connect();

    const listButton = await screen.findByRole("button", {
      name: /List Property for Sale/,
    });
    await waitFor(() => expect(listButton).toBeEnabled());
    await userEvent.click(listButton);

    // Dataset ₹3,00,00,000 at ₹1,00,000/ETH => 300 ETH; earnest fixed 10% => 30.
    await waitFor(() => expect(approve).toHaveBeenCalled());
    await waitFor(() =>
      expect(listForSaleWithRequirements).toHaveBeenCalledWith(
        1,
        ethers.utils.parseEther("300"),
        ethers.utils.parseEther("30"),
        true,
        true,
      ),
    );
    // The single authoritative price is displayed, from the dataset. No editable
    // price input exists anywhere in the listing form (covered by test A).
    expect(
      screen.getAllByText(/From MREID property dataset/).length,
    ).toBeGreaterThan(0);
    expect(screen.queryByLabelText(/INR asking price/)).toBeNull();
    contractSpy.mockRestore();
  });

  it("shows the buyer the same dataset price and a loan that is always price minus down payment (D, E, F, G)", async () => {
    connectWallet.mockResolvedValue({ ok: true, account: BUYER });
    getSaleStatus.mockResolvedValue(
      listedStatus({
        status: 2,
        statusName: "Under contract",
        buyer: BUYER,
        buyerFundedWei: ETH(30),
        lenderRequired: true,
        inspectionRequired: true,
        inspectionPassed: true,
      }),
    );

    render(
      <TransactionPanel mreidId="MREID_0000001" listedPriceInr={30000000} />,
    );
    await connect();

    expect(
      await screen.findByText(
        /Down payment \+ requested loan = property price/,
      ),
    ).toBeInTheDocument();
    await waitFor(() =>
      expect(
        screen.getByRole("button", { name: /Request financing/ }),
      ).toBeEnabled(),
    );
    // Choose 25% down payment: 25% of 300 = 75, loan = 225 (price minus down).
    await userEvent.selectOptions(screen.getByRole("combobox"), "2500");
    expect(screen.getAllByText("75 test ETH").length).toBeGreaterThan(0);
    expect(screen.getByText("225 test ETH")).toBeInTheDocument();
  });

  it("keeps a seller approval free of any price input and shows the purchase plan (H)", async () => {
    connectWallet.mockResolvedValue({ ok: true, account: SELLER });
    getSaleStatus.mockResolvedValue(
      listedStatus({
        status: 2,
        statusName: "Under contract",
        buyer: BUYER,
        buyerFundedWei: ETH(60),
        lenderFundedWei: ETH(240),
        sellerApproved: false,
        buyerApproved: true,
        inspectionRequired: true,
        inspectionPassed: true,
        lenderRequired: true,
        financing: noFinancing({
          requested: true,
          approved: true,
          downPaymentPctBps: 2000,
          downPaymentWei: ETH(60),
          loanAmountWei: ETH(240),
        }),
      }),
    );

    render(<TransactionPanel mreidId="MREID_0000001" />);
    await connect();

    expect(
      await screen.findByText(
        /Purchase plan - seller approval \(no price change\)/,
      ),
    ).toBeInTheDocument();
    await waitFor(() =>
      expect(
        screen.getByRole("button", { name: /Approve sale \(seller\)/ }),
      ).toBeEnabled(),
    );
    expect(screen.getByText("60 test ETH")).toBeInTheDocument();
    expect(screen.getByText("240 test ETH")).toBeInTheDocument();
    expect(
      screen.queryByLabelText(/asking price/),
    ).toBeNull();
    expect(
      screen.queryByRole("spinbutton", { name: /price/i }),
    ).toBeNull();
  });

  it("blocks a non-owner seller from seeing the listing action (J, K)", async () => {
    connectWallet.mockResolvedValue({ ok: true, account: LENDER });
    getSaleStatus.mockResolvedValue(
      listedStatus({ onOffer: false, status: 0, statusName: "Not listed" }),
    );
    demoRoleOf.mockImplementation((acc) =>
      String(acc).toLowerCase() === LENDER.toLowerCase() ? "lender" : null,
    );

    render(
      <TransactionPanel mreidId="MREID_0000001" listedPriceInr={30000000} />,
    );
    await connect();

    expect(
      await screen.findByText(/owned by another account/),
    ).toBeInTheDocument();
    expect(
      screen.queryByRole("button", { name: /List Property for Sale/ }),
    ).toBeNull();
    expect(screen.queryByLabelText(/INR asking price/)).toBeNull();
    expect(
      screen.queryByRole("button", { name: /Approve sale \(seller\)/ }),
    ).toBeNull();
  });

  it("shows No transaction workflow for an unregistered property (M)", async () => {
    connectWallet.mockResolvedValue({ ok: true, account: SELLER });
    getSaleStatus.mockResolvedValue({
      available: true,
      registered: false,
      supported: true,
      chainId: 31337,
    });

    render(
      <TransactionPanel mreidId="MREID_0009999" listedPriceInr={30000000} />,
    );
    await connect();

    expect(
      await screen.findByText(/No transaction workflow/),
    ).toBeInTheDocument();
    expect(
      screen.queryByRole("button", { name: /List Property for Sale/ }),
    ).toBeNull();
    expect(screen.queryByText(/INR asking price/)).toBeNull();
  });

  it("lets the seller close a listed sale", async () => {
    connectWallet.mockResolvedValue({ ok: true, account: SELLER });

    render(<TransactionPanel mreidId="MREID_0000001" />);
    await connect();

    expect(
      await screen.findByRole("button", { name: /Close listing \(removes from sale\)/ }),
    ).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: /Fund balance/ })).toBeNull();
    expect(screen.queryByRole("button", { name: /Approve sale/ })).toBeNull();
    expect(
      screen.queryByRole("button", { name: /Cancel & refund my test ETH/ }),
    ).toBeNull();
  });

  it("shows the buyer commit for an under-contract sale headed by the connected account", async () => {
    connectWallet.mockResolvedValue({ ok: true, account: BUYER });
    getSaleStatus.mockResolvedValue(
      listedStatus({
        status: 2,
        statusName: "Under contract",
        buyer: BUYER,
        buyerFundedWei: ETH(30),
      }),
    );

    render(<TransactionPanel mreidId="MREID_0000001" />);
    await connect();

    expect(
      await screen.findByRole("button", { name: /Fund balance \(270 test ETH\)/ }),
    ).toBeInTheDocument();
    expect(
      screen.getByRole("button", { name: /Approve sale \(buyer\)/ }),
    ).toBeInTheDocument();
    expect(
      screen.getByRole("button", { name: /Cancel & refund my test ETH/ }),
    ).toBeInTheDocument();
  });

  it("keeps a stranger's role cards read-only and points to the right account", async () => {
    connectWallet.mockResolvedValue({ ok: true, account: SELLER });
    getSaleStatus.mockResolvedValue(
      listedStatus({
        status: 2,
        statusName: "Under contract",
        buyer: BUYER,
        buyerFundedWei: ETH(30),
      }),
    );

    render(<TransactionPanel mreidId="MREID_0000001" />);
    await connect();

    expect(
      await screen.findByRole("button", { name: /Approve sale \(seller\)/ }),
    ).toBeEnabled();
    // The connected Seller is not the on-chain Buyer, so the Buyer card shows
    // no fund / approve / cancel buttons at all — just the account hint.
    expect(
      screen.queryByRole("button", { name: /Approve sale \(buyer\)/ }),
    ).toBeNull();
    expect(
      screen.queryByRole("button", { name: /Fund balance/ }),
    ).toBeNull();
    expect(
      screen.queryByRole("button", { name: /Cancel & refund my test ETH/ }),
    ).toBeNull();
    expect(
      screen.getByText(/Switch to the Buyer account to perform this action/),
    ).toBeInTheDocument();
  });

  it("reveals the finalize action when both parties approved and it is fully funded", async () => {
    connectWallet.mockResolvedValue({ ok: true, account: SELLER });
    getSaleStatus.mockResolvedValue(
      listedStatus({
        status: 3,
        statusName: "Approved",
        buyer: BUYER,
        buyerFundedWei: ETH(300),
        sellerApproved: true,
        buyerApproved: true,
      }),
    );

    render(<TransactionPanel mreidId="MREID_0000001" />);
    await connect();

    expect(
      await screen.findByRole("button", { name: /Finalize sale/ }),
    ).toBeInTheDocument();
    expect(
      screen.getByText(/All requirements met - ready to finalize/),
    ).toBeInTheDocument();
  });

  it("shows the finalized message once the sale is done", async () => {
    connectWallet.mockResolvedValue({ ok: true, account: SELLER });
    getSaleStatus.mockResolvedValue(
      listedStatus({
        status: 4,
        statusName: "Finalized",
        buyer: BUYER,
        buyerFundedWei: ETH(300),
        sellerApproved: true,
        buyerApproved: true,
      }),
    );

    render(<TransactionPanel mreidId="MREID_0000001" />);
    await connect();

    expect(
      await screen.findByText(/has been finalized on-chain/),
    ).toBeInTheDocument();
  });

  it("shows inspector verification actions to an assigned inspector", async () => {
    connectWallet.mockResolvedValue({ ok: true, account: INSPECTOR });
    getSaleStatus.mockResolvedValue(
      listedStatus({
        status: 2,
        statusName: "Under contract",
        buyer: BUYER,
        buyerFundedWei: ETH(30),
        inspectionRequired: true,
        inspectionPassed: false,
        accountRole: "inspector",
      }),
    );

    render(<TransactionPanel mreidId="MREID_0000001" />);
    await connect();

    expect(
      await screen.findByRole("button", { name: /Verify property \(inspection passed\)/ }),
    ).toBeInTheDocument();
    expect(
      screen.getByRole("button", { name: /Mark inspection failed/ }),
    ).toBeInTheDocument();
    expect(screen.getByText(/requires inspector verification/)).toBeInTheDocument();
  });

  it("hides inspector actions when the sale does not require inspection", async () => {
    connectWallet.mockResolvedValue({ ok: true, account: INSPECTOR });
    getSaleStatus.mockResolvedValue(
      listedStatus({
        status: 2,
        statusName: "Under contract",
        buyer: BUYER,
        buyerFundedWei: ETH(30),
        accountRole: "inspector",
      }),
    );

    render(<TransactionPanel mreidId="MREID_0000001" />);
    await connect();

    expect(
      screen.queryByRole("button", { name: /Verify property/ }),
    ).toBeNull();
    expect(
      screen.queryByRole("button", { name: /Mark inspection failed/ }),
    ).toBeNull();
  });

  it("lets the assigned lender approve or reject a pending financing request", async () => {
    connectWallet.mockResolvedValue({ ok: true, account: LENDER });
    getSaleStatus.mockResolvedValue(
      listedStatus({
        status: 2,
        statusName: "Under contract",
        buyer: BUYER,
        buyerFundedWei: ETH(30),
        lenderRequired: true,
        financing: noFinancing({
          requested: true,
          downPaymentPctBps: 2000,
          downPaymentWei: ETH(60),
          loanAmountWei: ETH(240),
        }),
        accountRole: "lender",
      }),
    );

    render(<TransactionPanel mreidId="MREID_0000001" />);
    await connect();

    expect(
      await screen.findByRole("button", { name: /Approve financing/ }),
    ).toBeInTheDocument();
    expect(
      screen.getByRole("button", { name: /Reject financing/ }),
    ).toBeInTheDocument();
    expect(
      screen.getByText(/the lender disburses the loan/),
    ).toBeInTheDocument();
  });

  it("shows the loan disbursal once the lender approves financing", async () => {
    connectWallet.mockResolvedValue({ ok: true, account: LENDER });
    getSaleStatus.mockResolvedValue(
      listedStatus({
        status: 2,
        statusName: "Under contract",
        buyer: BUYER,
        buyerFundedWei: ETH(30),
        lenderFundedWei: ETH(0),
        lenderRequired: true,
        financing: noFinancing({
          requested: true,
          approved: true,
          downPaymentPctBps: 2000,
          downPaymentWei: ETH(60),
          loanAmountWei: ETH(240),
        }),
        accountRole: "lender",
      }),
    );

    render(<TransactionPanel mreidId="MREID_0000001" />);
    await connect();

    expect(
      await screen.findByRole("button", { name: /Disburse loan \(240 test ETH\)/ }),
    ).toBeInTheDocument();
    expect(
      screen.queryByRole("button", { name: /Approve financing/ }),
    ).toBeNull();
  });

  it("keeps the buyer down-payment actions hidden for the lender role UI", async () => {
    connectWallet.mockResolvedValue({ ok: true, account: LENDER });
    getSaleStatus.mockResolvedValue(
      listedStatus({
        status: 2,
        statusName: "Under contract",
        buyer: BUYER,
        buyerFundedWei: ETH(30),
        lenderFundedWei: ETH(240),
        lenderRequired: true,
        financing: noFinancing({
          requested: true,
          approved: true,
          downPaymentPctBps: 2000,
          downPaymentWei: ETH(60),
          loanAmountWei: ETH(240),
        }),
        accountRole: "lender",
      }),
    );

    render(<TransactionPanel mreidId="MREID_0000001" />);
    await connect(LENDER);

    expect(await screen.findByText(/Loan fully disbursed/)).toBeInTheDocument();
    // The buyer's down-payment action belongs to the Buyer card, which is
    // read-only for the lender role.
    expect(
      screen.queryByRole("button", { name: /Pay down-payment balance/ }),
    ).toBeNull();
    expect(
      screen.getByText(/Switch to the Buyer account to perform this action/),
    ).toBeInTheDocument();
  });

  it("shows the buyer the financing request form on an extended sale", async () => {
    connectWallet.mockResolvedValue({ ok: true, account: BUYER });
    getSaleStatus.mockResolvedValue(
      listedStatus({
        status: 2,
        statusName: "Under contract",
        buyer: BUYER,
        buyerFundedWei: ETH(30),
        lenderRequired: true,
        inspectionRequired: true,
        inspectionPassed: true,
      }),
    );

    render(<TransactionPanel mreidId="MREID_0000001" />);
    await connect();

    expect(
      await screen.findByText(/choose your down payment/),
    ).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /Request financing/ })).toBeInTheDocument();
    expect(screen.getByText(/Estimated monthly EMI/)).toBeInTheDocument();
  });

  it("hides the buyer financing form while a request is awaiting the lender", async () => {
    connectWallet.mockResolvedValue({ ok: true, account: BUYER });
    getSaleStatus.mockResolvedValue(
      listedStatus({
        status: 2,
        statusName: "Under contract",
        buyer: BUYER,
        buyerFundedWei: ETH(30),
        lenderRequired: true,
        financing: noFinancing({
          requested: true,
          downPaymentPctBps: 2000,
          downPaymentWei: ETH(60),
          loanAmountWei: ETH(240),
        }),
      }),
    );

    render(<TransactionPanel mreidId="MREID_0000001" />);
    await connect();

    expect(
      screen.queryByRole("button", { name: /Request financing/ }),
    ).toBeNull();
    expect(screen.getByText(/pending lender decision/)).toBeInTheDocument();
  });

  it("lets the buyer pay the down-payment balance after financing is approved", async () => {
    connectWallet.mockResolvedValue({ ok: true, account: BUYER });
    getSaleStatus.mockResolvedValue(
      listedStatus({
        status: 2,
        statusName: "Under contract",
        buyer: BUYER,
        buyerFundedWei: ETH(30),
        lenderFundedWei: ETH(240),
        lenderRequired: true,
        financing: noFinancing({
          requested: true,
          approved: true,
          downPaymentPctBps: 2000,
          downPaymentWei: ETH(60),
          loanAmountWei: ETH(240),
        }),
      }),
    );

    render(<TransactionPanel mreidId="MREID_0000001" />);
    await connect();

    expect(
      await screen.findByRole("button", { name: /Pay down-payment balance \(30 test ETH\)/ }),
    ).toBeInTheDocument();
  });

  it("gates the seller approval on an extended sale until every condition passes", async () => {
    connectWallet.mockResolvedValue({ ok: true, account: SELLER });
    getSaleStatus.mockResolvedValue(
      listedStatus({
        status: 2,
        statusName: "Under contract",
        buyer: BUYER,
        buyerFundedWei: ETH(30),
        lenderFundedWei: ETH(240),
        sellerApproved: false,
        buyerApproved: false,
        inspectionRequired: true,
        inspectionPassed: true,
        lenderRequired: true,
        financing: noFinancing({
          requested: true,
          approved: true,
          downPaymentPctBps: 2000,
          downPaymentWei: ETH(60),
          loanAmountWei: ETH(240),
        }),
      }),
    );

    render(<TransactionPanel mreidId="MREID_0000001" />);
    await connect();

    expect(
      await screen.findByRole("button", { name: /Approve sale \(seller\)/ }),
    ).toBeDisabled();
    expect(
      screen.getByText(/buyer has not approved/),
    ).toBeInTheDocument();
  });

  it("enables the seller approval once every extended condition is satisfied", async () => {
    connectWallet.mockResolvedValue({ ok: true, account: SELLER });
    getSaleStatus.mockResolvedValue(
      listedStatus({
        status: 2,
        statusName: "Under contract",
        buyer: BUYER,
        buyerFundedWei: ETH(60),
        lenderFundedWei: ETH(240),
        sellerApproved: false,
        buyerApproved: true,
        inspectionRequired: true,
        inspectionPassed: true,
        lenderRequired: true,
        financing: noFinancing({
          requested: true,
          approved: true,
          downPaymentPctBps: 2000,
          downPaymentWei: ETH(60),
          loanAmountWei: ETH(240),
        }),
      }),
    );

    render(<TransactionPanel mreidId="MREID_0000001" />);
    await connect();

    expect(
      await screen.findByRole("button", { name: /Approve sale \(seller\)/ }),
    ).toBeEnabled();
  });

  it("blocks finalize until a required inspection passes", async () => {
    connectWallet.mockResolvedValue({ ok: true, account: SELLER });
    getSaleStatus.mockResolvedValue(
      listedStatus({
        status: 3,
        statusName: "Approved",
        buyer: BUYER,
        buyerFundedWei: ETH(300),
        sellerApproved: true,
        buyerApproved: true,
        inspectionRequired: true,
        inspectionPassed: false,
      }),
    );

    render(<TransactionPanel mreidId="MREID_0000001" />);
    await connect();

    expect(
      await screen.findByRole("button", { name: /Finalize sale/ }),
    ).toBeDisabled();
    expect(
      screen.getAllByText(/inspector has NOT passed inspection/).length,
    ).toBeGreaterThan(0);
    expect(screen.getByText(/awaiting inspector verdict/)).toBeInTheDocument();
  });

  it("reveals finalize once the required inspection has passed", async () => {
    connectWallet.mockResolvedValue({ ok: true, account: SELLER });
    getSaleStatus.mockResolvedValue(
      listedStatus({
        status: 3,
        statusName: "Approved",
        buyer: BUYER,
        buyerFundedWei: ETH(300),
        sellerApproved: true,
        buyerApproved: true,
        inspectionRequired: true,
        inspectionPassed: true,
      }),
    );

    render(<TransactionPanel mreidId="MREID_0000001" />);
    await connect();

    expect(
      await screen.findByRole("button", { name: /Finalize sale/ }),
    ).toBeInTheDocument();
    expect(
      screen.getByText(/Inspection passed - the seller may proceed/),
    ).toBeInTheDocument();
  });

  it("blocks finalize until required lender financing is approved and disbursed", async () => {
    connectWallet.mockResolvedValue({ ok: true, account: SELLER });
    getSaleStatus.mockResolvedValue(
      listedStatus({
        status: 3,
        statusName: "Approved",
        buyer: BUYER,
        buyerFundedWei: ETH(300),
        lenderFundedWei: ETH(0),
        sellerApproved: true,
        buyerApproved: true,
        lenderRequired: true,
        financing: noFinancing({
          requested: true,
          approved: true,
          downPaymentPctBps: 2000,
          downPaymentWei: ETH(60),
          loanAmountWei: ETH(240),
        }),
      }),
    );

    render(<TransactionPanel mreidId="MREID_0000001" />);
    await connect();

    expect(
      await screen.findByRole("button", { name: /Finalize sale/ }),
    ).toBeDisabled();
  });

  it("reveals finalize once required lender financing is approved and disbursed", async () => {
    connectWallet.mockResolvedValue({ ok: true, account: SELLER });
    getSaleStatus.mockResolvedValue(
      listedStatus({
        status: 3,
        statusName: "Approved",
        buyer: BUYER,
        buyerFundedWei: ETH(60),
        lenderFundedWei: ETH(240),
        sellerApproved: true,
        buyerApproved: true,
        lenderRequired: true,
        financing: noFinancing({
          requested: true,
          approved: true,
          downPaymentPctBps: 2000,
          downPaymentWei: ETH(60),
          loanAmountWei: ETH(240),
        }),
      }),
    );

    render(<TransactionPanel mreidId="MREID_0000001" />);
    await connect();

    expect(
      await screen.findByRole("button", { name: /Finalize sale/ }),
    ).toBeInTheDocument();
    expect(screen.getByText(/approved \(240 test ETH loan\)/)).toBeInTheDocument();
  });

  it("defaults the listing to the extended Inspector/Lender workflow", async () => {
    connectWallet.mockResolvedValue({ ok: true, account: SELLER });
    demoRoleOf.mockImplementation((acc) =>
      String(acc).toLowerCase() === SELLER.toLowerCase() ? "seller" : null,
    );
    getSaleStatus.mockResolvedValue(
      listedStatus({ onOffer: false, status: 0, statusName: "Not listed" }),
    );

    render(<TransactionPanel mreidId="MREID_0000001" />);
    await connect();

    const inspectorCheck = await screen.findByRole("checkbox", {
      name: /Require inspector verification/,
    });
    const lenderCheck = screen.getByRole("checkbox", {
      name: /Require lender financing/,
    });
    expect(inspectorCheck.checked).toBe(true);
    expect(lenderCheck.checked).toBe(true);
    await userEvent.click(inspectorCheck);
    expect(inspectorCheck.checked).toBe(false);
    expect(lenderCheck.checked).toBe(true);
    expect(
      screen.getByText(/extended Inspector\/Lender workflow runs by default/),
    ).toBeInTheDocument();
  });

  it("reflects the connected account's demo role chip", async () => {
    connectWallet.mockResolvedValue({ ok: true, account: INSPECTOR });
    getSaleStatus.mockResolvedValue(
      listedStatus({
        status: 2,
        statusName: "Under contract",
        buyer: BUYER,
        buyerFundedWei: ETH(30),
        inspectionRequired: true,
        accountRole: "inspector",
      }),
    );

    render(<TransactionPanel mreidId="MREID_0000001" />);
    await connect();

    expect(await screen.findByText(/Role: Inspector/)).toBeInTheDocument();
  });

  it("re-derives the workspace role when the wallet account changes, without a reload", async () => {
    connectWallet.mockResolvedValue({ ok: true, account: SELLER });
    getSaleStatus.mockImplementation((mreidId, account) =>
      Promise.resolve(
        listedStatus({
          status: 2,
          statusName: "Under contract",
          buyer: BUYER,
          buyerFundedWei: ETH(30),
          accountRole:
            (account &&
              String(account).toLowerCase() === SELLER.toLowerCase()) ||
            (account &&
              String(account).toLowerCase() === BUYER.toLowerCase())
              ? (String(account).toLowerCase() === SELLER.toLowerCase()
                  ? "seller"
                  : "buyer")
              : "viewer",
          sellerApproved: false,
          buyerApproved: false,
        }),
      ),
    );

    render(<TransactionPanel mreidId="MREID_0000001" />);
    await connect();

    expect(await screen.findByText(/Role: Seller/)).toBeInTheDocument();

    act(() => {
      mockSubscribeHandlers.onAccount(BUYER);
    });

    expect(await screen.findByText(/Role: Buyer/)).toBeInTheDocument();
    expect(
      screen.getByRole("button", { name: /Approve sale \(buyer\)/ }),
    ).toBeEnabled();
  });

  it("distinguishes NOT REQUIRED roles from unregistered properties", async () => {
    connectWallet.mockResolvedValue({ ok: true, account: INSPECTOR });
    getSaleStatus.mockResolvedValue(
      listedStatus({
        status: 2,
        statusName: "Under contract",
        buyer: BUYER,
        buyerFundedWei: ETH(30),
        accountRole: "inspector",
      }),
    );

    render(<TransactionPanel mreidId="MREID_0000001" />);
    await connect();

    expect(
      await screen.findByText(/different from a property that has not been registered/),
    ).toBeInTheDocument();
    expect(screen.getAllByText("NOT REQUIRED").length).toBeGreaterThan(0);
  });

  it("keeps a Model A sale free of extended requirements in the workspace", async () => {
    connectWallet.mockResolvedValue({ ok: true, account: SELLER });
    getSaleStatus.mockResolvedValue(
      listedStatus({
        status: 2,
        statusName: "Under contract",
        buyer: BUYER,
        buyerFundedWei: ETH(300),
        sellerApproved: false,
        buyerApproved: true,
      }),
    );

    render(<TransactionPanel mreidId="MREID_0000001" />);
    await connect();

    expect(
      await screen.findByText(/Model A — no inspection or lender required/),
    ).toBeInTheDocument();
    expect(screen.getAllByText("NOT REQUIRED").length).toBeGreaterThanOrEqual(2);
    expect(
      screen.queryByText(/requires inspector verification/),
    ).toBeNull();
    expect(screen.queryByText("awaiting inspector verdict")).toBeNull();
    expect(screen.queryByText(/not requested/)).toBeNull();
    expect(screen.queryByText(/approval waits until/)).toBeNull();
    expect(
      await screen.findByRole("button", { name: /Approve sale \(seller\)/ }),
    ).toBeEnabled();
  });

  it("shows the pending extended requirements on a Model B sale", async () => {
    connectWallet.mockResolvedValue({ ok: true, account: SELLER });
    getSaleStatus.mockResolvedValue(
      listedStatus({
        status: 2,
        statusName: "Under contract",
        buyer: BUYER,
        buyerFundedWei: ETH(30),
        inspectionRequired: true,
        inspectionPassed: false,
        lenderRequired: true,
      }),
    );

    render(<TransactionPanel mreidId="MREID_0000001" />);
    await connect();

    expect(
      await screen.findByText(/requires inspector verification/),
    ).toBeInTheDocument();
    expect(screen.getByText("awaiting inspector verdict")).toBeInTheDocument();
    expect(screen.getByText(/not requested/)).toBeInTheDocument();
    expect(screen.queryByText(/Model A — no inspection/)).toBeNull();
  });

  it("shows the stored interest rate in percent (850 bps as 8.5%)", async () => {
    connectWallet.mockResolvedValue({ ok: true, account: LENDER });
    getSaleStatus.mockResolvedValue(
      listedStatus({
        status: 2,
        statusName: "Under contract",
        buyer: BUYER,
        buyerFundedWei: ETH(30),
        lenderFundedWei: ETH(240),
        lenderRequired: true,
        financing: noFinancing({
          requested: true,
          approved: true,
          downPaymentPctBps: 2000,
          downPaymentWei: ETH(60),
          loanAmountWei: ETH(240),
          interestRateBps: 850,
          loanTenureMonths: 240,
        }),
        accountRole: "lender",
      }),
    );

    render(<TransactionPanel mreidId="MREID_0000001" />);
    await connect();

    expect(
      await screen.findByText(/approved \(240 test ETH loan\) @ 8.5%/),
    ).toBeInTheDocument();
    expect(
      screen.getByText(
        /Request terms: 20% down payment, 8.5% annual interest, 240 months/,
      ),
    ).toBeInTheDocument();
  });

  it("defaults the financing interest input to 8.5 percent", async () => {
    connectWallet.mockResolvedValue({ ok: true, account: BUYER });
    getSaleStatus.mockResolvedValue(
      listedStatus({
        status: 2,
        statusName: "Under contract",
        buyer: BUYER,
        buyerFundedWei: ETH(30),
        lenderRequired: true,
        inspectionRequired: true,
        inspectionPassed: true,
      }),
    );

    render(<TransactionPanel mreidId="MREID_0000001" />);
    await connect();

    const interest = await screen.findByDisplayValue("8.5");
    expect(interest).toBeInTheDocument();
    expect(interest.getAttribute("value")).toBe("8.5");
  });

  it("submits the interest rate to the escrow in basis points (8.5% = 850)", async () => {
    connectWallet.mockResolvedValue({ ok: true, account: BUYER });
    getSigner.mockResolvedValue({});
    getSaleStatus.mockResolvedValue(
      listedStatus({
        status: 2,
        statusName: "Under contract",
        buyer: BUYER,
        buyerFundedWei: ETH(30),
        lenderRequired: true,
        inspectionRequired: true,
        inspectionPassed: true,
      }),
    );

    const requestFinancing = jest
      .fn()
      .mockResolvedValue({ wait: jest.fn().mockResolvedValue(undefined) });
    const contractSpy = jest
      .spyOn(ethers, "Contract")
      .mockReturnValue({ requestFinancing });

    render(<TransactionPanel mreidId="MREID_0000001" />);
    await connect();

    const requestButton = await screen.findByRole("button", {
      name: /Request financing/,
    });
    await waitFor(() => expect(requestButton).toBeEnabled());
    await userEvent.click(requestButton);

    await waitFor(() =>
      expect(requestFinancing).toHaveBeenCalledWith(1, 2000, 850, 240),
    );
    contractSpy.mockRestore();
  });

  it("shows the unregistered workspace for MREID_0000006 with no fabricated sale state", async () => {
    getSaleStatus.mockResolvedValue({
      available: true,
      registered: false,
      supported: true,
      chainId: 31337,
    });

    render(<TransactionPanel mreidId="MREID_0000006" />);

    expect(
      await screen.findByText(/MREID_0000006 is not yet tokenized/),
    ).toBeInTheDocument();
    expect(screen.queryByText("Sale summary")).toBeNull();
    expect(screen.queryByText("Purchase Progress")).toBeNull();
    expect(screen.queryByRole("button", { name: /Finalize sale/ })).toBeNull();
    expect(screen.queryByText(/Model A — no inspection/)).toBeNull();
    expect(screen.queryByText("300 test ETH")).toBeNull();
  });

  it("shows No active sale for a registered property that has never been listed", async () => {
    connectWallet.mockResolvedValue({ ok: true, account: SELLER });
    demoRoleOf.mockImplementation((acc) =>
      String(acc).toLowerCase() === SELLER.toLowerCase() ? "seller" : null,
    );
    getSaleStatus.mockResolvedValue(
      listedStatus({
        onOffer: false,
        status: 0,
        statusName: "Not listed",
        seller: ZERO,
      }),
    );

    render(<TransactionPanel mreidId="MREID_0000001" />);
    await connect();

    expect(
      await screen.findByText(
        /No active sale\. This property is registered on chain/,
      ),
    ).toBeInTheDocument();
    expect(
      screen.getByText(/progress appears here once the seller lists/),
    ).toBeInTheDocument();
    // No fabricated 0-ETH price / earnest / funded figures.
    expect(screen.queryByText("0 test ETH")).toBeNull();
    expect(screen.queryByRole("button", { name: /Commit & deposit/ })).toBeNull();
    expect(screen.queryByText(/Model A — no inspection/)).toBeNull();
  });

  it("shows a tokenized-unlisted property as owned by the Seller and available to be listed", async () => {
    demoRoleOf.mockImplementation((acc) =>
      String(acc).toLowerCase() === SELLER.toLowerCase() ? "seller" : null,
    );
    getSaleStatus.mockResolvedValue(
      listedStatus({
        onOffer: false,
        status: 0,
        statusName: "Not listed",
        seller: ZERO,
      }),
    );

    render(<TransactionPanel mreidId="MREID_0000001" />);

    expect(
      await screen.findByText(/No active sale\./),
    ).toBeInTheDocument();
    expect(
      screen.getByText(
        /Property NFT is owned by the Seller and is available to be listed\./,
      ),
    ).toBeInTheDocument();
    // The full dataset is tokenized now: never a "not yet tokenized" workspace
    // for a registered property, and no fabricated 20% / purchase-plan figures.
    expect(
      screen.queryByText(/MREID_0000001 is not yet tokenized/),
    ).toBeNull();
    expect(screen.queryByText(/No transaction workflow/)).toBeNull();
    expect(screen.queryByText(/Model A — no inspection/)).toBeNull();
  });

  it("keeps the Seller card read-only when the demo seller does not own the NFT", async () => {
    connectWallet.mockResolvedValue({ ok: true, account: SELLER });
    demoRoleOf.mockImplementation((acc) =>
      String(acc).toLowerCase() === SELLER.toLowerCase() ? "seller" : null,
    );
    getSaleStatus.mockResolvedValue(
      listedStatus({
        onOffer: false,
        status: 0,
        statusName: "Not listed",
        seller: ZERO,
        ownerOf: REGISTRAR,
      }),
    );

    render(<TransactionPanel mreidId="MREID_0000001" />);
    await connect(SELLER);

    // The demo badge must not override ownership: no list form, read-only card.
    expect(
      screen.queryByRole("button", { name: /List Property for Sale/ }),
    ).toBeNull();
    expect(
      await screen.findByText(/owned by another account/),
    ).toBeInTheDocument();
    expect(
      screen.getByText(/Switch to the Seller account to perform this action/),
    ).toBeInTheDocument();
    expect(screen.queryByText("Action available for you")).toBeNull();
  });

  it("enables the demo buyer commit on an open listing", async () => {
    connectWallet.mockResolvedValue({ ok: true, account: BUYER });
    demoRoleOf.mockImplementation((acc) =>
      String(acc).toLowerCase() === BUYER.toLowerCase() ? "buyer" : null,
    );

    render(<TransactionPanel mreidId="MREID_0000001" />);
    await connect(BUYER);

    const commit = await screen.findByRole("button", {
      name: /Commit & deposit earnest 30 test ETH/,
    });
    expect(commit).toBeEnabled();
    expect(screen.getByText("Role: Buyer")).toBeInTheDocument();
  });

  it("keeps the commit action hidden for a stranger on an open listing", async () => {
    connectWallet.mockResolvedValue({
      ok: true,
      account: "0x15d34AAf54267DB7D7c367839AAf71A00a2C6A65",
    });

    render(<TransactionPanel mreidId="MREID_0000001" />);
    await connect("0x15d34AAf54267DB7D7c367839AAf71A00a2C6A65");

    expect(
      await screen.findByText(
        /Switch to the Buyer account to perform this action/,
      ),
    ).toBeInTheDocument();
    // A stranger has no Buyer workspace, so there is no commit button at all.
    expect(
      screen.queryByRole("button", { name: /Commit & deposit/ }),
    ).toBeNull();
    expect(screen.getByText("Role: Viewer")).toBeInTheDocument();
  });

  it("re-derives the workspace role instantly when the account switches on an open listing", async () => {
    connectWallet.mockResolvedValue({ ok: true, account: SELLER });
    demoRoleOf.mockImplementation((account) => {
      const a = String(account).toLowerCase();
      if (a === SELLER.toLowerCase()) return "seller";
      if (a === BUYER.toLowerCase()) return "buyer";
      return null;
    });
    getSaleStatus.mockImplementation((mreidId, account) =>
      Promise.resolve(
        listedStatus({
          status: 1,
          statusName: "Listed for sale",
          accountRole:
            String(account).toLowerCase() === SELLER.toLowerCase()
              ? "seller"
              : "viewer",
        }),
      ),
    );

    render(<TransactionPanel mreidId="MREID_0000001" />);
    await connect(SELLER);

    expect(await screen.findByText(/Role: Seller/)).toBeInTheDocument();

    act(() => {
      mockSubscribeHandlers.onAccount(BUYER);
    });

    await waitFor(() =>
      expect(screen.getByText(/Role: Buyer/)).toBeInTheDocument(),
    );
    expect(
      await screen.findByRole("button", {
        name: /Commit & deposit earnest 30 test ETH/,
      }),
    ).toBeEnabled();
  });
});