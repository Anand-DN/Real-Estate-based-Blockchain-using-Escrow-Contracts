import { render, screen, fireEvent, waitFor, within, act } from "@testing-library/react";
import { ethers } from "ethers";
import App from "./App";
import config from "./config.json";

// Demo-role accounts from scripts/millowSeed.json (see src/config.json).
const BUYER = "0xf39Fd6e51aad88F6F4ce6aB8827279cffFb92266";
const SELLER = "0x70997970C51812dc3A010C7d01b50e0d17dc79C8";
const INSPECTOR = "0x3C44CdDdB6a900fa2b585dd299e03d12FA4293BC";
const LENDER = "0x90F79bf6EB2c4f870365E785982E1f101E93b906";

// Live browser-equivalent integration test.  Requires:
//   - Hardhat node on http://127.0.0.1:8545 (chain 31337) with the MILLOW layer
//     deployed (scripts/deployMillow.js) and the full 29,135-property dataset
//     seeded as NFTs at the Seller (scripts/tokenizeAllMreid.js),
//   - backend API on http://localhost:8001 (npm run valuation).
// It renders the REAL App and drives the actual read path used by the browser.

jest.setTimeout(120000);

describe("Marketplace live blockchain UI", () => {
  let node;

  beforeAll(async () => {
    // Bypass the test-env network skip so the components hit the live node.
    process.env.REACT_APP_BLOCKCHAIN_RPC_URL = "http://127.0.0.1:8545";
    // Real browsers run the app at http://localhost:3000. jsdom enforces a fake
    // CORS policy and ships an incomplete `performance`, so route window.fetch
    // through Node's raw http/https stack (same wire protocol as the browser,
    // no CORS/undici interference) against the backend and the node.
    const nodeHttp = require("http");
    const nodeHttps = require("https");
    const readBody = (res) =>
      new Promise((resolve, reject) => {
        let data = "";
        res.setEncoding("utf8");
        res.on("data", (chunk) => (data += chunk));
        res.on("end", () => resolve(data));
        res.on("error", reject);
      });
    window.fetch = async (input, init = {}) => {
      const url = typeof input === "string" ? input : input.url;
      const u = new URL(url);
      const transport = u.protocol === "https:" ? nodeHttps : nodeHttp;
      const body =
        init.body != null
          ? typeof init.body === "string"
            ? init.body
            : JSON.stringify(init.body)
          : undefined;
      const res = await new Promise((resolve, reject) => {
        const req = transport.request(
          {
            host: u.hostname,
            port: u.port || (transport === nodeHttps ? 443 : 80),
            path: u.pathname + u.search,
            method: init.method || "GET",
            headers: {
              accept: "application/json",
              ...(init.headers || {}),
              ...(body != null
                ? { "content-type": "application/json" }
                : {}),
            },
          },
          resolve
        );
        req.on("error", reject);
        if (body != null) req.write(body);
        req.end();
      });
      const text = await readBody(res);
      return {
        ok: res.statusCode >= 200 && res.statusCode < 300,
        status: res.statusCode,
        json: async () => JSON.parse(text),
        text: async () => text,
      };
    };
    node = new ethers.providers.JsonRpcProvider("http://127.0.0.1:8545");
    const listeners = {};
    let currentAccount = null;
    window.ethereum = {
      request: async (args) => {
        if (args.method === "eth_accounts") {
          return currentAccount ? [currentAccount] : [];
        }
        if (args.method === "eth_requestAccounts") {
          if (!currentAccount) {
            const accounts = await node.send("eth_accounts", []);
            if (accounts && accounts.length) currentAccount = accounts[0];
          }
          return currentAccount ? [currentAccount] : [];
        }
        return node.send(args.method, args.params || []);
      },
      // Test helper: simulate the user switching the active MetaMask account.
      _switchAccount: (address) => {
        currentAccount = address;
        (listeners.accountsChanged || []).forEach((fn) =>
          fn([address]),
        );
        return Promise.resolve([address]);
      },
      // Test helper: reset the wallet back to the disconnected state so the
      // shared closure cannot leak an account between tests in this suite.
      _reset: () => {
        currentAccount = null;
        Object.keys(listeners).forEach((key) => delete listeners[key]);
        return Promise.resolve();
      },
      on: (event, fn) => {
        (listeners[event] = listeners[event] || []).push(fn);
      },
      removeListener: (event, fn) => {
        listeners[event] = (listeners[event] || []).filter(
          (f) => f !== fn,
        );
      },
    };
  });

  afterAll(() => {
    delete process.env.REACT_APP_BLOCKCHAIN_RPC_URL;
  });

  beforeEach(async () => {
    await window.ethereum._reset();
  });

  const openProperty = async (mreidId, locationPattern) => {
    const card = await screen.findByRole(
      "button",
      { name: locationPattern },
      { timeout: 30000 }
    );
    fireEvent.click(card);
    const dialog = await screen.findByRole("dialog", { name: "Property details" });
    await within(dialog).findByText("Blockchain & NFT Status");
    // Wait for the live chain status grid to finish rendering.
    await waitFor(
      () => {
        expect(within(dialog).queryByText("Checking the local blockchain…")).not.toBeInTheDocument();
      },
      { timeout: 30000 }
    );
    return dialog;
  };

  it("MREID_0000001 is the seeded token #1: Seller-owned, not listed, no active sale", async () => {
    render(<App />);

    fireEvent.click(screen.getByRole("button", { name: "Marketplace" }));

    const dialog = await openProperty("MREID_0000001", /JP Nagar Phase 1/);

    // Blockchain Status: chain, NFT address, token id, owner, not-listed
    // (no registry listing, no escrow sale), mint tx.  The full dataset is
    // seeded on the prototype chain, so MREID_0000001 is token #1 with the
    // NFT at the Seller and no listing — never "not yet tokenized".
    expect(within(dialog).getByText("Millow Localhost (Hardhat Network)")).toBeInTheDocument();
    expect(within(dialog).getByText("31337")).toBeInTheDocument();
    expect(
      within(dialog).getByText(
        new RegExp(config["31337"].propertyNft.address.slice(0, 10)),
      ),
    ).toBeInTheDocument();
    expect(within(dialog).getByText("#1")).toBeInTheDocument();
    expect(within(dialog).getByText("Not listed")).toBeInTheDocument();
    expect(within(dialog).getByText(/(Seller)/)).toBeInTheDocument();
    expect(within(dialog).getByText("Mint transaction")).toBeInTheDocument();
    expect(within(dialog).queryByText("On offer")).toBeNull();

    // The exact error strings from the regression must be GONE.
    expect(within(dialog).queryByText(/Unsupported network/)).not.toBeInTheDocument();
    expect(
      within(dialog).queryByText(/not registered on the prototype blockchain/),
    ).not.toBeInTheDocument();

    // Transaction Workspace: token 1 has no escrow sale yet, so the workspace
    // must NOT fabricate price/earnest/funded figures — it says so explicitly.
    const workspace = within(dialog).getByText("Transaction Workspace").closest("section");
    expect(workspace).not.toBeNull();
    await waitFor(
      () => {
        expect(
          within(workspace).getByText(
            /No active sale\. This property is registered on chain/,
          ),
        ).toBeInTheDocument();
      },
      { timeout: 30000 }
    );
    expect(within(workspace).getByText("Not in a sale")).toBeInTheDocument();
    expect(
      within(workspace).getByText(
        "Property NFT is owned by the Seller and is available to be listed.",
      ),
    ).toBeInTheDocument();
    expect(within(workspace).queryByText("300 test ETH")).toBeNull();
    expect(within(workspace).queryByText("30 test ETH")).toBeNull();
    expect(within(workspace).queryByText(/Model A — no inspection/)).toBeNull();
    expect(
      within(workspace).queryByRole("button", { name: /Commit & deposit/ }),
    ).toBeNull();
    expect(
      within(workspace).queryByText(/is not yet tokenized/),
    ).toBeNull();

    // Connect as the Seller (the NFT owner): the Seller card becomes
    // actionable and offers the List action for the unlisted property, with
    // no active escrow sale yet.
    await act(async () => {
      await window.ethereum._switchAccount(SELLER);
    });
    await waitFor(
      () => {
        expect(
          within(workspace).getByText("Role: Seller"),
        ).toBeInTheDocument();
      },
      { timeout: 30000 }
    );
    expect(
      within(workspace).getByText("List this property for sale"),
    ).toBeInTheDocument();
    expect(
      within(workspace).getByRole("button", { name: "List Property for Sale" }),
    ).toBeInTheDocument();
    expect(
      within(workspace).queryByText(
        "This registered property is not on offer and is owned by another account.",
      ),
    ).toBeNull();

    // Sale History: the mint event is on chain (registered), so the mint row
    // is visible rather than a fabricated ledger or the empty-state copy.
    const history = within(dialog).getByText("Sale History").closest("section");
    await waitFor(
      () => {
        expect(
          within(history).getByText("NFT minted (registered)"),
        ).toBeInTheDocument();
      },
      { timeout: 30000 }
    );
    expect(
      within(history).queryByText("This property has not been through an escrow sale yet."),
    ).toBeNull();
    expect(within(history).queryByText("Listed for sale")).toBeNull();
  });

  it("MREID_0000002 is the seeded token #2: Seller-owned, not listed, no active sale", async () => {
    const meta = await (await fetch("http://localhost:8001/api/properties/MREID_0000002")).json();
    const pattern = new RegExp(meta.property.location);

    render(<App />);
    fireEvent.click(screen.getByRole("button", { name: "Marketplace" }));

    const dialog = await openProperty("MREID_0000002", pattern);

    // Registered as token 2 with the NFT at the Seller and no listing.
    expect(within(dialog).getByText("#2")).toBeInTheDocument();
    expect(within(dialog).getByText("Not listed")).toBeInTheDocument();
    expect(within(dialog).getByText(/(Seller)/)).toBeInTheDocument();
    expect(within(dialog).queryByText("On offer")).toBeNull();

    // Transaction Workspace: no escrow sale exists, so the workspace says so
    // and never fabricates price/earnest/funding figures.
    const workspace = within(dialog).getByText("Transaction Workspace").closest("section");
    await waitFor(
      () => {
        expect(
          within(workspace).getByText(/No active sale\./),
        ).toBeInTheDocument();
        expect(
          within(workspace).getByText(
            "Property NFT is owned by the Seller and is available to be listed.",
          ),
        ).toBeInTheDocument();
      },
      { timeout: 30000 }
    );
    expect(within(workspace).queryByText(/awaiting inspector verdict/)).toBeNull();
    expect(within(workspace).queryByText("300 test ETH")).toBeNull();
    expect(within(workspace).queryByText("30 test ETH")).toBeNull();

    // The Seller can list this property for sale (extended workflow default):
    // connect as the NFT owner so the Seller card becomes actionable.
    await act(async () => {
      await window.ethereum._switchAccount(SELLER);
    });
    await waitFor(
      () => {
        expect(
          within(workspace).getByText("Role: Seller"),
        ).toBeInTheDocument();
      },
      { timeout: 30000 }
    );
    expect(
      within(workspace).getByText("List this property for sale"),
    ).toBeInTheDocument();
    expect(
      within(workspace).getByRole("button", { name: "List Property for Sale" }),
    ).toBeInTheDocument();

    // Sale History: only the mint row — no escrow events exist on chain.
    const history = within(dialog).getByText("Sale History").closest("section");
    await waitFor(
      () => {
        expect(
          within(history).getByText("NFT minted (registered)"),
        ).toBeInTheDocument();
      },
      { timeout: 30000 }
    );
    expect(within(history).queryByText("Listed for sale")).toBeNull();
  });

  it("MREID_0000008 is tokenized, unchanged listing: Seller-owned and not listed", async () => {
    const meta = await (
      await fetch("http://localhost:8001/api/properties/MREID_0000008")
    ).json();
    const pattern = new RegExp(meta.property.location);

    render(<App />);
    fireEvent.click(screen.getByRole("button", { name: "Marketplace" }));

    const dialog = await openProperty("MREID_0000008", pattern);

    // Blockchain Status: the full dataset is tokenized, so this MREID is
    // registered on chain with its NFT at the Seller and no listing.
    await waitFor(
      () => {
        expect(within(dialog).queryByText("Not registered")).toBeNull();
      },
      { timeout: 30000 }
    );
    expect(
      within(dialog).getAllByText("MREID_0000008").length,
    ).toBeGreaterThan(0);
    expect(within(dialog).getByText("Not listed")).toBeInTheDocument();
    // Owner row names the demo role next to the owner address.
    expect(within(dialog).getByText(/(Seller)/)).toBeInTheDocument();

    // The Transaction Workspace shows the tokenized-unlisted state: no "not
    // yet tokenized" placeholder, and a clear "available to be listed" hint.
    const workspace = within(dialog)
      .getByText("Transaction Workspace")
      .closest("section");
    await waitFor(
      () => {
        expect(
          within(workspace).getByText(/No active sale\./),
        ).toBeInTheDocument();
        expect(
          within(workspace).getByText(
            /Property NFT is owned by the Seller and is available to be listed\./,
          ),
        ).toBeInTheDocument();
      },
      { timeout: 30000 }
    );
    expect(
      within(workspace).queryByText(/is not yet tokenized/),
    ).toBeNull();
    expect(
      within(workspace).queryByText(/No transaction workflow/),
    ).toBeNull();

    // The Review reading must NOT have been replaced by the read-only notice.
    expect(
      within(dialog).queryByText(/read(\s|-)?only/),
    ).not.toBeInTheDocument();

    // The simplified workflow has no register flow: no register card, no
    // Registrar account prompt, no MINTER plumbing anywhere in the UI.
    expect(
      within(dialog).queryByText(/Register property NFT/),
    ).toBeNull();
    expect(
      within(dialog).queryAllByText(/Registrar/).length,
    ).toBe(0);
    expect(
      within(dialog).queryByText(/MINTER_ROLE/),
    ).toBeNull();

    // Sale History: the property was minted on chain (registered), so the
    // mint row is visible rather than the unregistered empty-state copy.
    const history = within(dialog).getByText("Sale History").closest("section");
    await waitFor(
      () => {
        expect(
          within(history).getByText("NFT minted (registered)"),
        ).toBeInTheDocument();
      },
      { timeout: 30000 }
    );
    expect(
      within(history).queryByText(/not tokenized on the prototype chain/),
    ).toBeNull();
  });

  it("re-derives the workspace role when the user switches accounts on a tokenized unlisted property, live", async () => {
    const meta = await (
      await fetch("http://localhost:8001/api/properties/MREID_0000002")
    ).json();
    const pattern = new RegExp(meta.property.location);

    render(<App />);
    fireEvent.click(screen.getByRole("button", { name: "Marketplace" }));

    const dialog = await openProperty("MREID_0000002", pattern);
    const workspace = within(dialog)
      .getByText("Transaction Workspace")
      .closest("section");

    // The property is tokenized but NOT on offer: no Buyer commit candidate
    // exists, so the demo buyer account derives to Viewer with no actionable
    // card (the listing-only commit gate never fabricates a Buyer role).
    const connectButton = await within(workspace).findByRole("button", {
      name: /Connect wallet \(any Hardhat account\)/,
    });
    fireEvent.click(connectButton);
    await waitFor(
      () => {
        expect(
          within(workspace).getByText("Role: Viewer"),
        ).toBeInTheDocument();
      },
      { timeout: 30000 }
    );
    expect(
      within(workspace).queryAllByText("Action available for you").length,
    ).toBe(0);

    // Switch to the demo inspector: role chip re-derives without any reload,
    // and only the Inspector card (on-chain INSPECTOR_ROLE) stays actionable.
    await act(async () => {
      await window.ethereum._switchAccount(INSPECTOR);
    });
    await waitFor(
      () => {
        expect(
          within(workspace).getByText("Role: Inspector"),
        ).toBeInTheDocument();
      },
      { timeout: 30000 }
    );
    await waitFor(
      () => {
        expect(
          within(workspace).getAllByText("Action available for you").length,
        ).toBe(1);
        expect(
          within(workspace).getAllByText(
            /Switch to the Buyer account to perform this action/,
          ).length,
        ).toBeGreaterThan(0);
      },
      { timeout: 30000 }
    );
  });

  it("maps every demo account to its on-chain workspace role on a tokenized unlisted property", async () => {
    const meta = await (
      await fetch("http://localhost:8001/api/properties/MREID_0000002")
    ).json();
    const pattern = new RegExp(meta.property.location);

    render(<App />);
    fireEvent.click(screen.getByRole("button", { name: "Marketplace" }));

    const dialog = await openProperty("MREID_0000002", pattern);
    const workspace = within(dialog)
      .getByText("Transaction Workspace")
      .closest("section");

    const SELLER = "0x70997970C51812dc3A010C7d01b50e0d17dc79C8";
    const BUYER_ACC = "0xf39Fd6e51aad88F6F4ce6aB8827279cffFb92266";
    const LENDER = "0x90F79bf6EB2c4f870365E785982E1f101E93b906";
    const STRANGER = "0x15d34AAf54267DB7D7c367839AAf71A00a2C6A65";

    // The property is tokenized but has no active listing or escrow sale —
    // the workspace must never fabricate an on-offer / inspection state.
    await waitFor(
      () => {
        expect(
          within(workspace).getByText(/No active sale\./),
        ).toBeInTheDocument();
        expect(
          within(workspace).queryByText(/awaiting inspector verdict/),
        ).toBeNull();
      },
      { timeout: 30000 }
    );

    const switchTo = async (address, expectedRole, activeCards) => {
      await act(async () => {
        await window.ethereum._switchAccount(address);
      });
      await waitFor(
        () => {
          expect(
            within(workspace).getByText(`Role: ${expectedRole}`),
          ).toBeInTheDocument();
        },
        { timeout: 30000 }
      );
      await waitFor(
        () => {
          expect(
            within(workspace).queryAllByText("Action available for you").length,
          ).toBe(activeCards);
        },
        { timeout: 30000 }
      );
    };

    // The demo seller owns the NFT on this chain: Seller card stays actionable.
    await switchTo(SELLER, "Seller", 1);
    // No listing exists, so the demo buyer has nothing to commit to: Viewer.
    await switchTo(BUYER_ACC, "Viewer", 0);
    // The on-chain INSPECTOR_ROLE keeps the Inspector card actionable.
    await switchTo("0x3C44CdDdB6a900fa2b585dd299e03d12FA4293BC", "Inspector", 1);
    // The on-chain LENDER_ROLE keeps the Lender card actionable.
    await switchTo(LENDER, "Lender", 1);
    // The former Registrar (chain MINTER holder) is no longer an application
    // role: the account drops to Viewer with no active card at all.  This
    // proves the Registrar concept has been removed from the UI.
    await switchTo("0xBcd4042DE499D14e55001CcbB24a551F3b954096", "Viewer", 0);
    // A stranger resolves to Viewer with no role card actionable at all.
    await switchTo(STRANGER, "Viewer", 0);

    // No fabricated sale state: no Model A chip, no "awaiting inspector verdict".
    expect(
      within(workspace).queryByText(/Model A — no inspection/),
    ).toBeNull();
    expect(
      within(workspace).queryByText(/awaiting inspector verdict/),
    ).toBeNull();
  });
});