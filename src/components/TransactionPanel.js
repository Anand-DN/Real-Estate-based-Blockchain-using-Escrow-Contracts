import { useCallback, useEffect, useMemo, useState } from "react";
import { ethers } from "ethers";
import config from "../config.json";
import MillowEscrow from "../abis/MillowEscrow.json";
import PropertyNFT from "../abis/PropertyNFT.json";
import {
  getSaleStatus,
  getPropertyChainStatus,
  demoRoleOf,
  ROLE_NAMES,
} from "../lib/blockchain";
import { demoRateInrPerEth, demoRateLabel, inrToWei } from "../lib/demoRate";
import { formatEth, formatInr, formatInrCompact } from "../lib/format";
import { connectWallet, getSigner, hasWallet, subscribeWallet } from "../lib/wallet";
import SaleHistory from "./SaleHistory";

const STATUS_TEXT = {
  0: "Not in a sale",
  1: "Listed for sale",
  2: "Under contract",
  3: "Approved - ready to finalize",
  4: "Finalized",
  5: "Cancelled",
};

const DOWN_PAYMENT_OPTIONS_BPS = [1000, 1500, 2000, 2500, 3000, 4000, 5000];

// The MILLOW workflow has ONE authoritative price: the property's existing
// MREID dataset price.  The seller never types a price.  The earnest the
// buyer commits is a fixed share of that dataset price (10%, matching the
// prototype's demo convention), never a seller-entered figure.
const LIST_EARNEST_BPS = 1000; // 10% of the dataset price

const ZERO = "0x0000000000000000000000000000000000000000";

const ready = (value) =>
  value && !Number.isNaN(Number(value)) && Number(value) > 0;

const bn = (value) => (value ? ethers.BigNumber.from(value) : ethers.BigNumber.from(0));

// Convert on-chain test-ETH (a Wei BigNumber or number) to its INR equivalent
// at the demo conversion rate, so dashboard summaries can show compact INR.
const weiToInr = (value) =>
  Number(ethers.utils.formatEther(bn(value))) * demoRateInrPerEth;

// When the connected account is not the role that performs an action, the
// action stays visible but disabled, with a pointer to the right account.
const performerHint = (roleKey, connected) =>
  connected
    ? `Switch to the ${ROLE_NAMES[roleKey]} account to perform this action.`
    : "Connect your wallet (any Hardhat account) to perform this action.";

const RoleCard = ({ title, status, connected, active, children, footer }) => (
  <div className={`tx-card tx-role-card${active ? " tx-role-card--active" : ""}`}>
    <div className="tx-card-head">
      <h4>{title}</h4>
      <span className="tx-role-state">
        {active ? "Action available for you" : "Read-only for you"}
      </span>
    </div>
    {status && <p className="tx-hint">{status}</p>}
    <div className="tx-role-actions">{children}</div>
    {footer && (
      <p className="tx-hint tx-lock-hint" role="status">
        {footer}
      </p>
    )}
  </div>
);

export default function TransactionPanel({
  mreidId,
  chainRecord,
  listedPriceInr,
  onSettled,
  refreshTick,
}) {
  const [status, setStatus] = useState(null);
  const [account, setAccount] = useState(null);
  const [pending, setPending] = useState("");
  const [notice, setNotice] = useState("");
  // The list form defaults to the extended Inspector/Lender workflow: the
  // demonstration property is approved for financing (payment Model B), so the
  // buyer commits earnest, the inspector verifies, the lender approves and
  // disburses the loan, and only then does the sale settle.
  const [inspectionReq, setInspectionReq] = useState(true);
  const [lenderReq, setLenderReq] = useState(true);
  const [dpBps, setDpBps] = useState(2000);
  // The input is a percentage for humans; it is converted to basis points
  // (bps = percentage x 100) when the financing request is submitted.
  const [ratePct, setRatePct] = useState("8.5");
  const [tenureMonths, setTenureMonths] = useState("240");

  const refresh = useCallback(async () => {
    try {
      const [sale] = await Promise.all([
        getSaleStatus(mreidId, account, chainRecord),
        getPropertyChainStatus(mreidId, chainRecord),
      ]);
      setStatus(sale);
    } catch {
      setStatus({ available: false, error: "Blockchain unavailable." });
    }
  }, [mreidId, account, chainRecord]);

  useEffect(() => {
    refresh();
  }, [refresh]);

  useEffect(() => {
    if (refreshTick > 0) refresh();
  }, [refreshTick, refresh]);

  useEffect(
    () => subscribeWallet({ onAccount: setAccount, onNetwork: refresh }),
    [refresh],
  );

  const chainId = status && status.supported ? status.chainId : null;

  const escrowAddress = useMemo(() => {
    const cfg = config[String(chainId)];
    return cfg && cfg.millowEscrow ? cfg.millowEscrow.address : null;
  }, [chainId]);

  const nftAddress = useMemo(() => {
    const cfg = config[String(chainId)];
    return cfg && cfg.propertyNft ? cfg.propertyNft.address : null;
  }, [chainId]);

  const isMe = useCallback(
    (addr) => account && addr && addr.toLowerCase() === account.toLowerCase(),
    [account],
  );

  const accountRole = (status && status.accountRole) || null;
  // On-chain facts decide who may act — the demo badge never overrides actual
  // ownership.  The seller is the escrow's sale seller or the real NFT owner;
  // the buyer is the account committed on the escrow sale.
  const amSeller = !!(
    status &&
    status.registered &&
    (isMe(status.seller) || isMe(status.ownerOf))
  );
  const amBuyer = !!(
    status &&
    status.registered &&
    status.buyer &&
    isMe(status.buyer)
  );
  const amOwner = !!(status && status.ownerOf && isMe(status.ownerOf));
  const statusNum = status && status.registered ? Number(status.status ?? 0) : null;
  const listed = statusNum === 1;
  const underContract = statusNum === 2 || statusNum === 3;
  const isFinalized = statusNum === 4;
  const isCancelled = statusNum === 5;
  // No escrow sale has started yet (registered, possibly listed on the
  // marketplace registry, but no on-chain price/earnest/funding exists).  The
  // workspace must not fabricate sale figures for such a property.
  const noActiveSale = statusNum === 0;
  const buyerPresent = !!(
    status &&
    status.buyer &&
    status.buyer.toLowerCase() !== ZERO
  );
  // Only the demo buyer may commit while a listing is open; once a buyer is
  // committed, the buyer actions belong to the on-chain committed account.
  const commitCandidate = !!(
    account &&
    listed &&
    !buyerPresent &&
    demoRoleOf(account) === "buyer"
  );
  // The demo inspector/lender accounts hold INSPECTOR_ROLE / LENDER_ROLE on the
  // escrow, so accountRole already resolves them from on-chain grants.  Demo
  // identities only fill the badge when no on-chain fact applies.
  const businessRole = commitCandidate
    ? "buyer"
    : accountRole || demoRoleOf(account) || null;
  const amInspector = accountRole === "inspector";
  const amLender = accountRole === "lender";
  const roleName = (businessRole && ROLE_NAMES[businessRole]) || "Viewer";
  const canList =
    status &&
    status.registered &&
    status.onOffer === false &&
    amOwner;

  const sellerWorkspaceActive = !!(account && amSeller);
  const buyerWorkspaceActive = !!(account && (amBuyer || commitCandidate));
  const inspectorWorkspaceActive = !!(account && amInspector);
  const lenderWorkspaceActive = !!(account && amLender);

  const fin = (status && status.financing) || {};
  const financingRequested = !!fin.requested;
  const financingApproved = !!fin.approved;
  const financingRejected = !!fin.rejected;
  const isExtended = !!(status && (status.inspectionRequired || status.lenderRequired));

  const priceWei = bn(status && status.priceWei);
  const buyerFunded = bn(status && status.buyerFundedWei);
  const lenderFunded = bn(status && status.lenderFundedWei);
  const totalFunded = buyerFunded.add(lenderFunded);
  const downPaymentWei = bn(fin.downPaymentWei);
  const loanAmountWei = bn(fin.loanAmountWei);
  const fullFunding = priceWei.gt(0) && totalFunded.eq(priceWei);
  // Authoritative listing price: the MREID dataset price (INR), converted to
  // test-ETH for the on-chain listing.  The seller never sets a price — the
  // dataset is the single source of truth.  When the dataset price is not
  // available (e.g. a pure on-chain fixture) the on-chain listing's price is
  // the fallback so the flow still uses one authoritative figure.
  const listingInr = ready(listedPriceInr);
  const listingPriceWei = listingInr
    ? inrToWei(listedPriceInr)
    : priceWei.gt(0)
      ? priceWei
      : bn(status && status.earnestWei).mul(10000).div(LIST_EARNEST_BPS);
  const listingEarnestWei = priceWei.gt(0)
    ? priceWei.mul(LIST_EARNEST_BPS).div(10000)
    : listingPriceWei.mul(LIST_EARNEST_BPS).div(10000);

  const loanFullyFunded = financingApproved && lenderFunded.eq(loanAmountWei);
  const buyerDownComplete = financingApproved && buyerFunded.eq(downPaymentWei);
  // Purchase-plan preview: the down payment is a share of the asking price and
  // the requested loan is always exactly the remainder, so the two can never
  // sum to more or less than the property price.
  const downPaymentPreview = priceWei.gt(0)
    ? priceWei.mul(Number(dpBps)).div(10000)
    : ethers.BigNumber.from(0);
  const loanPreview = priceWei.gt(0)
    ? priceWei.sub(downPaymentPreview)
    : ethers.BigNumber.from(0);

  const sellerCanApproveExtended =
    isExtended &&
    (!status.inspectionRequired || status.inspectionPassed) &&
    (!status.lenderRequired || loanFullyFunded) &&
    status.buyerApproved &&
    fullFunding;

  const sellerApprovalHint = isExtended && !sellerCanApproveExtended
    ? [
        (!status.inspectionRequired || status.inspectionPassed)
          ? null
          : "inspector has NOT passed inspection",
        !status.lenderRequired
          ? null
          : financingApproved && loanFullyFunded
            ? null
            : "loan is not approved and fully disbursed",
        status.buyerApproved ? null : "buyer has not approved",
        fullFunding ? null : "escrow is not fully funded",
      ].filter(Boolean)
    : [];

  const finalizable =
    statusNum === 3 &&
    status.buyerApproved &&
    status.sellerApproved &&
    fullFunding &&
    (!status.inspectionRequired || status.inspectionPassed) &&
    (!status.lenderRequired || (financingApproved && loanFullyFunded));

  const finalizeBlockers = [];
  if (statusNum === 3 && !finalizable) {
    if (!status.buyerApproved) finalizeBlockers.push("buyer has not approved");
    if (!status.sellerApproved) finalizeBlockers.push("seller has not approved");
    if (priceWei.gt(0) && !fullFunding) finalizeBlockers.push("escrow is not fully funded");
    if (status.inspectionRequired && !status.inspectionPassed) {
      finalizeBlockers.push("inspector has NOT passed inspection");
    }
    if (status.lenderRequired && !(financingApproved && loanFullyFunded)) {
      finalizeBlockers.push("loan is not approved and fully disbursed");
    }
  }

  // Estimated EMI, purely front-end: P * r * (1+r)^n / ((1+r)^n - 1), with the
  // annual rate in percent split to a monthly rate. Not a real quote.
  let emiEth = 0;
  if (ready(ratePct) && ready(tenureMonths) && priceWei.gt(0)) {
    const loanWeiPreview = priceWei
      .mul(10000 - Number(dpBps))
      .div(10000);
    const loanEth = Number(ethers.utils.formatEther(loanWeiPreview));
    const annualRate = (Number(ratePct) || 0) / 100;
    const monthlyRate = annualRate / 12;
    const n = Number(tenureMonths || 0);
    if (monthlyRate > 0 && n > 0) {
      const factor = Math.pow(1 + monthlyRate, n);
      emiEth = (loanEth * monthlyRate * factor) / (factor - 1);
    } else if (n > 0) {
      emiEth = loanEth / n;
    }
  }

  const run = async (label, fn) => {
    try {
      setNotice("");
      setPending(label);
      const result = await fn();
      if (result && !result.ok) {
        setNotice(`Transaction failed: ${result.error}`);
        return;
      }
      setNotice((result && result.message) || "Transaction confirmed.");
      await refresh();
      if (onSettled) onSettled();
    } catch (err) {
      setNotice(
        `Transaction failed: ${err && err.reason ? err.reason : err.message || "unknown error"}`,
      );
    } finally {
      setPending("");
    }
  };

  const handleConnect = async () => {
    const result = await connectWallet();
    if (result.ok) {
      setAccount(result.account);
    } else {
      setNotice(result.error);
    }
  };

  const withContract = async (fn) => {
    const signer = await getSigner();
    if (!signer) return { ok: false, error: "No Web3 wallet detected." };
    const escrow = new ethers.Contract(escrowAddress, MillowEscrow, signer);
    return fn(escrow, signer);
  };

  const handleList = () => {
    // The listing price and earnest are ALWAYS derived from the MREID dataset
    // price — the seller never enters them.  The panel keeps no price input at
    // all, so there is nothing here to validate other than the on-chain guard.
    if (!nftAddress || !escrowAddress || !status || !status.tokenId) return;
    if (!listingPriceWei.gt(0) || !listingEarnestWei.gt(0)) {
      setNotice("The property dataset price is unavailable; cannot list.");
      return;
    }
    if (listingEarnestWei.gte(listingPriceWei)) {
      setNotice("Earnest must be less than the full price.");
      return;
    }
    run("listing", async () => {
      const signer = await getSigner();
      if (!signer) return { ok: false, error: "No Web3 wallet detected." };
      const nft = new ethers.Contract(nftAddress, PropertyNFT, signer);
      const escrow = new ethers.Contract(escrowAddress, MillowEscrow, signer);
      const approved = await nft.getApproved(status.tokenId);
      if (approved.toLowerCase() !== escrowAddress.toLowerCase()) {
        const approveTx = await nft.approve(escrowAddress, status.tokenId);
        await approveTx.wait();
      }
      // The demonstration listing uses the extended Inspector/Lender workflow
      // by default; unchecking both requirements sends the plain Model A sale.
      const tx =
        inspectionReq || lenderReq
          ? await escrow.listForSaleWithRequirements(
              status.tokenId,
              listingPriceWei,
              listingEarnestWei,
              inspectionReq,
              lenderReq,
            )
          : await escrow.listForSale(
              status.tokenId,
              listingPriceWei,
              listingEarnestWei,
            );
      await tx.wait();
    });
  };

  const handleClose = () =>
    run("closing", () =>
      withContract(async (escrow) => {
        const tx = await escrow.closeListing(status.tokenId);
        await tx.wait();
      }),
    );

  const handleCommit = () =>
    run("committing", () =>
      withContract((escrow) =>
        escrow.commitAndDeposit(status.tokenId, { value: status.earnestWei }),
      ),
    );

  const handleFund = () =>
    run("funding", () =>
      withContract(async (escrow) => {
        const owed = priceWei.sub(totalFunded);
        const tx = await escrow.fundBalance(status.tokenId, { value: owed });
        await tx.wait();
      }),
    );

  const handlePayDownPayment = () =>
    run("paying down payment", () =>
      withContract(async (escrow) => {
        const owed = downPaymentWei.sub(buyerFunded);
        const tx = await escrow.payDownPayment(status.tokenId, { value: owed });
        await tx.wait();
      }),
    );

  const handleApproveBuyer = () =>
    run("approving", () =>
      withContract(async (escrow) => {
        const tx = await escrow.approveBuyer(status.tokenId);
        await tx.wait();
      }),
    );

  const handleApproveSeller = () =>
    run("approving", () =>
      withContract(async (escrow) => {
        const tx = await escrow.approveSeller(status.tokenId);
        await tx.wait();
      }),
    );

  const handleFinalize = () =>
    run("finalizing", () =>
      withContract(async (escrow) => {
        const tx = await escrow.finalizeSale(status.tokenId);
        await tx.wait();
      }),
    );

  const handleCancel = () =>
    run("cancelling", () =>
      withContract(async (escrow) => {
        const tx = await escrow.cancelSale(status.tokenId);
        await tx.wait();
      }),
    );

  const handleInspect = (passed) =>
    run("inspecting", () =>
      withContract(async (escrow) => {
        const tx = await escrow.setInspectionStatus(status.tokenId, passed);
        await tx.wait();
      }),
    );

  const handleRequestFinancing = () => {
    if (!DOWN_PAYMENT_OPTIONS_BPS.includes(Number(dpBps))) {
      setNotice("Choose a valid down-payment percentage.");
      return;
    }
    const rate = Number(ratePct);
    const tenure = Number(tenureMonths);
    if (!rate || rate <= 0 || rate > 50) {
      setNotice("Enter an annual interest rate between 0.01% and 50%.");
      return;
    }
    if (!tenure || tenure <= 0) {
      setNotice("Enter a loan tenure in months.");
      return;
    }
    run("requesting financing", () =>
      withContract(async (escrow) => {
        // UI works in percent; the escrow contract stores basis points.
        const interestRateBps = Math.round(rate * 100);
        const tx = await escrow.requestFinancing(
          status.tokenId,
          Number(dpBps),
          interestRateBps,
          tenure,
        );
        await tx.wait();
      }),
    );
  };

  const handleFinancingDecision = (approve) =>
    run(approve ? "approving financing" : "rejecting financing", () =>
      withContract(async (escrow) => {
        const tx = approve
          ? await escrow.approveFinancing(status.tokenId)
          : await escrow.rejectFinancing(status.tokenId);
        await tx.wait();
      }),
    );

  const handleFundLoan = () =>
    run("disbursing loan", () =>
      withContract(async (escrow) => {
        const owed = loanAmountWei.sub(lenderFunded);
        const tx = await escrow.fundLoan(status.tokenId, { value: owed });
        await tx.wait();
      }),
    );

  if (!status || !status.available) {
    return (
      <section className="tx-panel">
        <h3>Transaction Workspace</h3>
        <p className="tx-unavailable">
          {status && status.error ? status.error : "Blockchain unavailable."}
        </p>
      </section>
    );
  }

  if (!status.supported) {
    return (
      <section className="tx-panel">
        <h3>Transaction Workspace</h3>
        <p className="tx-unavailable">
          This property demo runs on the Hardhat chain 31337. The connected
          chain ({status.chainId}) is not supported for transactions.
        </p>
      </section>
    );
  }

  const accountCard = (
    <div className="tx-card tx-account-card">
      <div className="tx-card-head">
        <h4>Connected account</h4>
        {account && (
          <span className="tx-role">Role: {roleName}</span>
        )}
      </div>
      {!account ? (
        <>
          <div className="tx-account-actions">
            <button
              type="button"
              className="tx-action"
              onClick={handleConnect}
              disabled={!!pending}
            >
              {pending ? "Working…" : "Connect wallet (any Hardhat account)"}
            </button>
          </div>
          <p className="tx-hint">
            Use a Hardhat test account with test ETH. Import private keys from
            the running node into MetaMask; no real funds are used.
          </p>
        </>
      ) : (
        <>
          <div className="tx-account">
            <span className="tx-account-address">{account}</span>
            {hasWallet() && (
              <button type="button" className="tx-action tx-action--small" onClick={handleConnect}>
                Switch account
              </button>
            )}
          </div>
          <p className="tx-hint">
            Account changes apply to this workspace instantly — no page reload.
          </p>
        </>
      )}
    </div>
  );

  if (status.tokenizationUnknown) {
    return (
      <section className="tx-panel">
        <h3>Transaction Workspace</h3>
        <div className="tx-note">
          <strong>Token status unavailable.</strong> The indexed MREID mapping
          and the connected blockchain do not currently confirm the same
          token, so no transaction workflow is offered.
        </div>
        {status.tokenId ? (
          <p className="tx-hint">Indexed token ID: #{status.tokenId}</p>
        ) : null}
        {status.error ? <p className="tx-hint tx-warn">{status.error}</p> : null}
        {accountCard}
      </section>
    );
  }

  // ------------------------------------------------------------------
  // Unregistered property: only MREIDs outside the tokenized dataset hit
  // this branch now (the full 29,135-property dataset is seeded on chain).
  // The simplified MILLOW workflow starts from a property that is already
  // on the prototype chain with its NFT at the seller, so there is nothing
  // to act on here.  No registration action is offered.
  // ------------------------------------------------------------------
  if (status.registered === false) {
    return (
      <section className="tx-panel">
        <h3>Transaction Workspace</h3>
        <div className="tx-legal tx-legal--compact">
          Prototype on {demoRateLabel} ({"test ETH on Hardhat chain 31337"}). No
          real money is involved. Blockchain records are a technical
          proof-of-concept; legal ownership remains subject to applicable Indian
          property and registration law.
        </div>

        <div className="tx-note">
          <strong>{mreidId} is not yet tokenized.</strong> This property exists
          in the MILLOW dataset but has no NFT on the prototype chain, so it has
          no listing, escrow, or on-chain history yet.
        </div>

        {accountCard}

        <div className="tx-card">
          <h4>No transaction workflow</h4>
          <p className="tx-hint">
            The simplified MILLOW workflow starts from a property whose NFT is
            already with the seller. This MREID record is not on the prototype
            chain, so there is no listing, financing or settlement to act on
            here — and no registration step in the demo.
          </p>
          <p className="tx-hint">
            Pre-minted demo properties (e.g. MREID_0000001 and MREID_0000002)
            show the full Seller → Buyer → Inspector → Lender → settlement
            workflow.
          </p>
        </div>
      </section>
    );
  }

  if (status.workflowAvailable === false) {
    return (
      <section className="tx-panel">
        <h3>Transaction Workspace</h3>
        <div className="tx-note">
          <strong>NFT found; transaction workflow unavailable.</strong> The
          connected chain does not have the MILLOW escrow deployment required
          for listing, financing, or settlement actions.
        </div>
        {accountCard}
      </section>
    );
  }

  const priceStr = formatEth(priceWei);
  const earnestStr = formatEth(status.earnestWei);
  const finRatePct =
    fin.interestRateBps && Number(fin.interestRateBps) > 0
      ? Number(fin.interestRateBps) / 100
      : null;

  const hasSale = !!(
    status &&
    status.seller &&
    status.seller.toLowerCase() !== ZERO
  );

  const priceInr = weiToInr(priceWei);
  // Purchase plan figures: once financing is approved the escrow's locked split
  // is authoritative; before that the summary previews the buyer's selected
  // down-payment share (down + loan always equal the price exactly).
  const downWei = financingApproved ? downPaymentWei : downPaymentPreview;
  const loanWei = financingApproved ? loanAmountWei : loanPreview;
  const downInr = weiToInr(downWei);
  const loanInr = weiToInr(loanWei);
  const downPct = financingRequested && fin.downPaymentPctBps
    ? Number(fin.downPaymentPctBps) / 100
    : Number(dpBps) / 100;

  // The buyer controls the payment plan: either a single full payment or a
  // down payment plus loan that always sum to the asking price.
  const financeMode = !!status.lenderRequired;
  const stages = financeMode
    ? [
        {
          label: "Seller lists property",
          done: hasSale,
          detail: hasSale ? "Listed" : "Not yet",
        },
        {
          label: "Buyer chooses payment plan",
          done: !!financingRequested,
          detail: financingRequested ? "Requested" : "Not yet",
        },
        {
          label: "Property inspection",
          done: !status.inspectionRequired || !!status.inspectionPassed,
          neutral: !status.inspectionRequired,
          detail: !status.inspectionRequired
            ? "Not required"
            : status.inspectionPassed
              ? "Passed"
              : "Awaiting verdict",
        },
        {
          label: "Loan approval",
          done: financingApproved && loanFullyFunded,
          detail: !financingRequested
            ? "Awaiting request"
            : financingApproved
              ? loanFullyFunded
                ? "Approved & disbursed"
                : "Approved - not disbursed"
              : financingRejected
                ? "Rejected"
                : "Awaiting lender",
        },
        {
          label: "Seller approval",
          done: !!status.sellerApproved,
          detail: status.sellerApproved ? "Approved" : "Not yet",
        },
        {
          label: "Ownership transfer",
          done: isFinalized,
          detail: isFinalized ? "NFT to buyer" : "Pending finalize",
        },
      ]
    : [
        {
          label: "Seller lists property",
          done: hasSale,
          detail: hasSale ? "Listed" : "Not yet",
        },
        {
          label: "Buyer pays in full",
          done: fullFunding,
          detail: fullFunding
            ? `Funded ${formatEth(priceWei)}`
            : `${formatEth(totalFunded)} of ${formatEth(priceWei)}`,
        },
        {
          label: "Property inspection",
          done: !status.inspectionRequired || !!status.inspectionPassed,
          neutral: !status.inspectionRequired,
          detail: !status.inspectionRequired
            ? "Not required"
            : status.inspectionPassed
              ? "Passed"
              : "Awaiting verdict",
        },
        {
          label: "Seller approval",
          done: !!status.sellerApproved,
          detail: status.sellerApproved ? "Approved" : "Not yet",
        },
        {
          label: "Ownership transfer",
          done: isFinalized,
          detail: isFinalized ? "NFT to buyer" : "Pending finalize",
        },
      ];

  const pendingStage = stages.find((stage) => !stage.done && !stage.neutral);
  const progressNote = isFinalized
    ? "Sale finalized - the NFT was delivered to the buyer and the price paid to the seller."
    : statusNum === 3
      ? finalizable
        ? "All requirements met - ready to finalize."
        : `Waiting on: ${finalizeBlockers.join(", ")}.`
      : pendingStage
        ? `Next step: ${pendingStage.label}.`
        : null;

  return (
    <section className="tx-panel">
      <h3>Transaction Workspace</h3>
      <div className="tx-legal tx-legal--compact">
        Prototype on {demoRateLabel} ({"test ETH on Hardhat chain 31337"}). No
        real money is involved. Blockchain records are a technical
        proof-of-concept; legal ownership remains subject to applicable Indian
        property and registration law.
      </div>

      {/* Sale Summary */}
      <div className="tx-card tx-summary-card">
        <h4>Sale summary</h4>
        {noActiveSale ? (
          <>
            <div className="tx-summary-grid">
              <div className="tx-stat">
                <span className="tx-stat-label">Status</span>
                <strong className="tx-stat-value">
                  {STATUS_TEXT[statusNum] || statusNum}
                </strong>
              </div>
            </div>
            <p className="tx-hint tx-warn" role="status">
              No active sale. This property is registered on chain but has no
              escrow sale yet — price, earnest and funded figures appear here
              once the seller lists it.
            </p>
            {listingInr ? (
              <p className="tx-hint">
                Property dataset price:{" "}
                <strong>{formatInr(listedPriceInr)}</strong>{" "}
                ({formatEth(listingPriceWei)}) — the sale price when the seller
                lists this property.
              </p>
            ) : null}
            {status.ownerOf && demoRoleOf(status.ownerOf) === "seller" && (
              <p className="tx-hint">
                Property NFT is owned by the Seller and is available to be
                listed.
              </p>
            )}
          </>
        ) : (
          <>
            <div className="tx-summary-grid">
              <div className="tx-stat">
                <span className="tx-stat-label">Price</span>
                <strong className="tx-stat-value">
                  {formatInrCompact(priceInr)}
                </strong>
                <span className="tx-hint-inline">{priceStr}</span>
              </div>
              <div className="tx-stat">
                <span className="tx-stat-label">Status</span>
                <strong className="tx-stat-value">
                  {STATUS_TEXT[statusNum] || statusNum}
                </strong>
              </div>
            </div>
            <div className="tx-plan">
              <div className="tx-plan-title">Payment plan</div>
              {financeMode ? (
                <>
                  <div className="tx-plan-row">
                    <span className="tx-label">
                      Down payment ({downPct}% of price)
                    </span>
                    <strong>{formatInrCompact(downInr)}</strong>
                  </div>
                  <div className="tx-plan-row">
                    <span className="tx-label">Requested loan (balance)</span>
                    <strong>{formatInrCompact(loanInr)}</strong>
                  </div>
                </>
              ) : (
                <>
                  <div className="tx-plan-row">
                    <span className="tx-label">Payment</span>
                    <strong>Full payment — no loan</strong>
                  </div>
                  <div className="tx-plan-row">
                    <span className="tx-label">Buyer earnest</span>
                    <strong>{earnestStr}</strong>
                  </div>
                </>
              )}
            </div>
            <div className="tx-summary-chips">
              {isExtended ? (
                <>
                  {status.inspectionRequired && (
                    <span className="tx-chip">
                      <span className="tx-label">Inspection:</span>{" "}
                      {status.inspectionPassed
                        ? "passed"
                        : "awaiting inspector verdict"}
                    </span>
                  )}
                  {status.lenderRequired && (
                    <span className="tx-chip">
                      <span className="tx-label">Financing:</span>{" "}
                      {!financingRequested
                        ? "not requested"
                        : financingApproved
                          ? `approved (${formatEth(loanAmountWei)} loan)${finRatePct ? ` @ ${finRatePct}%` : ""}`
                          : financingRejected
                            ? "rejected - buyer may re-request"
                            : "pending lender decision"}
                    </span>
                  )}
                </>
              ) : (
                <span className="tx-chip tx-chip--model-a">
                  <span className="tx-label">Requirements:</span> Model A — no
                  inspection or lender required
                </span>
              )}
            </div>
          </>
        )}
      </div>

      {accountCard}

      {/* Role workspace */}
      <div className="tx-roles">
      {/* SELLER */}
      <RoleCard
        title="Seller"
        active={sellerWorkspaceActive}
        connected={!!account}
        footer={!sellerWorkspaceActive && performerHint("seller", !!account)}
      >
        {canList && !listed && !underContract && !isFinalized && !isCancelled && (
          <fieldset className="tx-fieldset">
            <legend>List this property for sale</legend>
            <div className="tx-list-price">
              <div className="tx-list-price-label">Sale price</div>
              <div className="tx-list-price-value">
                {listingInr
                  ? formatInr(listedPriceInr)
                  : formatEth(listingPriceWei)}
              </div>
              <div className="tx-list-price-sub">
                {listingInr
                  ? `From MREID property dataset · ${formatEth(listingPriceWei)}`
                  : "From the MREID property dataset"}
              </div>
            </div>
            <div className="tx-list-price">
              <div className="tx-list-price-label">Buyer commitment (earnest)</div>
              <div className="tx-list-price-value">
                {formatEth(listingEarnestWei)}
              </div>
              <div className="tx-list-price-sub">
                Fixed {Number(LIST_EARNEST_BPS) / 100}% of the dataset price —
                not set by the seller
              </div>
            </div>
            <label className="tx-check">
              <input
                type="checkbox"
                checked={inspectionReq}
                disabled={!sellerWorkspaceActive}
                onChange={(event) => setInspectionReq(event.target.checked)}
              />
              Require inspector verification
            </label>
            <label className="tx-check">
              <input
                type="checkbox"
                checked={lenderReq}
                disabled={!sellerWorkspaceActive}
                onChange={(event) => setLenderReq(event.target.checked)}
              />
              Require lender financing
            </label>
            <button
              type="button"
              className="tx-action"
              onClick={handleList}
              disabled={!sellerWorkspaceActive || !!pending}
            >
              {pending ? "Working…" : "List Property for Sale"}
            </button>
            <p className="tx-legal">
              The sale price is the property&#39;s existing MREID dataset price —
              there is no editable price. The extended Inspector/Lender workflow
              runs by default (the demonstration property is approved for
              financing): the inspector verifies the property, the buyer
              requests financing, the lender approves and disburses the loan,
              and the buyer pays the down payment. Unchecking both requirements
              lists the plain Buyer/Seller sale (Payment Model A). Demo
              conversion {demoRateLabel}{" "}
              turns the INR price into test ETH — no real money.
            </p>
          </fieldset>
        )}

        {account && listed && sellerWorkspaceActive && (
          <div className="tx-account-actions">
            <button
              type="button"
              className="tx-action"
              onClick={handleClose}
              disabled={!!pending}
            >
              {pending ? "Working…" : "Close listing (removes from sale)"}
            </button>
          </div>
        )}

        {account && underContract && sellerWorkspaceActive && (
          <div className="tx-account-actions">
            {isExtended ? (
              <>
                <div className="tx-plan">
                  <div className="tx-plan-title">
                    Purchase plan - seller approval (no price change)
                  </div>
                  <div className="tx-plan-row">
                    <span className="tx-label">Property price</span>
                    <strong>{priceStr}</strong>
                  </div>
                  <div className="tx-plan-row">
                    <span className="tx-label">Down payment</span>
                    <strong>{formatEth(downPaymentWei)}</strong>
                  </div>
                  <div className="tx-plan-row">
                    <span className="tx-label">Requested loan</span>
                    <strong>{formatEth(loanAmountWei)}</strong>
                  </div>
                  <p className="tx-hint tx-plan-check">
                    Down payment + requested loan = property price:{" "}
                    {formatEth(downPaymentWei)} + {formatEth(loanAmountWei)}{" "}
                    = {priceStr}. The seller approves this exact plan — the
                    price is the property&#39;s dataset price and cannot be
                    changed here.
                  </p>
                </div>
                {!status.sellerApproved ? (
                  <button
                    type="button"
                    className="tx-action"
                    onClick={handleApproveSeller}
                    disabled={!sellerWorkspaceActive || !!pending || !sellerCanApproveExtended}
                  >
                    {pending ? "Working…" : "Approve sale (seller)"}
                  </button>
                ) : (
                  <span className="tx-hint">Seller approved - waiting to finalize.</span>
                )}
                {sellerApprovalHint.length > 0 && (
                  <p className="tx-hint tx-warn">
                    Seller approval waits until:{" "}
                    {sellerApprovalHint.join(", ")}.
                  </p>
                )}
              </>
            ) : (
              <button
                type="button"
                className="tx-action"
                onClick={handleApproveSeller}
                disabled={!sellerWorkspaceActive || !!pending}
              >
                {pending ? "Working…" : "Approve sale (seller)"}
              </button>
            )}
          </div>
        )}

        {statusNum === 0 && !canList && (
          <p className="tx-hint">
            This registered property is not on offer and is owned by another
            account.
          </p>
        )}
      </RoleCard>

      {/* BUYER */}
      <RoleCard
        title="Buyer"
        active={buyerWorkspaceActive}
        connected={!!account}
        footer={!buyerWorkspaceActive && performerHint("buyer", !!account)}
      >
        {listed && buyerWorkspaceActive && (
          <div className="tx-account-actions">
            <button
              type="button"
              className="tx-action"
              onClick={handleCommit}
              disabled={!!pending}
            >
              {pending
                ? "Working…"
                : `Commit & deposit earnest ${earnestStr}`}
            </button>
            <p className="tx-hint">
              The escrow requires exactly {earnestStr} of test ETH to start the
              sale. You need the Hardhat account&#39;s private key; this is
              prototype money.
            </p>
          </div>
        )}

        {underContract && !isExtended && buyerWorkspaceActive && (
          <div className="tx-account-actions">
            {totalFunded.lt(priceWei) && (
              <button
                type="button"
                className="tx-action"
                onClick={handleFund}
                disabled={!buyerWorkspaceActive || !!pending}
              >
                {pending
                  ? "Working…"
                  : `Fund balance (${formatEth(priceWei.sub(totalFunded))})`}
              </button>
            )}
            {!status.buyerApproved && (
              <button
                type="button"
                className="tx-action"
                onClick={handleApproveBuyer}
                disabled={!buyerWorkspaceActive || !!pending}
              >
                {pending ? "Working…" : "Approve sale (buyer)"}
              </button>
            )}
            {status.buyerApproved && (
              <span className="tx-hint">Buyer approval recorded.</span>
            )}
            {(totalFunded.gt(0) || buyerFunded.gt(0)) && (
              <button
                type="button"
                className="tx-action tx-danger"
                onClick={handleCancel}
                disabled={!buyerWorkspaceActive || !!pending}
              >
                {pending ? "Working…" : "Cancel & refund my test ETH"}
              </button>
            )}
          </div>
        )}

        {underContract && isExtended && buyerWorkspaceActive && (
          <div className="tx-account-actions">
            {status.lenderRequired &&
              !financingApproved &&
              (!financingRequested || financingRejected) && (
                <fieldset className="tx-fieldset">
                  <legend>Financing request - choose your down payment</legend>
                  {status.inspectionRequired && !status.inspectionPassed && (
                    <p className="tx-hint">
                      Financing can only be requested once the inspector has passed
                      the property.
                    </p>
                  )}
                  <div className="tx-row">
                    <span className="tx-label">Down payment</span>
                    <select
                      value={Number(dpBps)}
                      disabled={!buyerWorkspaceActive}
                      onChange={(event) => setDpBps(Number(event.target.value))}
                    >
                      {DOWN_PAYMENT_OPTIONS_BPS.map((bps) => (
                        <option key={bps} value={bps}>
                          {bps / 100}%
                        </option>
                      ))}
                    </select>
                  </div>
                  <div className="tx-row">
                    <span className="tx-label">Annual interest rate (%)</span>
                    <input
                      type="number"
                      min="0.01"
                      max="50"
                      step="0.05"
                      value={ratePct}
                      disabled={!buyerWorkspaceActive}
                      onChange={(event) => setRatePct(event.target.value)}
                    />
                  </div>
                  <div className="tx-row">
                    <span className="tx-label">Tenure (months)</span>
                    <input
                      type="number"
                      min="1"
                      value={tenureMonths}
                      disabled={!buyerWorkspaceActive}
                      onChange={(event) => setTenureMonths(event.target.value)}
                    />
                  </div>
                  <div className="tx-plan">
                    <div className="tx-plan-title">
                      Purchase plan (down payment + loan = price)
                    </div>
                    <div className="tx-plan-row">
                      <span className="tx-label">Property price</span>
                      <strong>{priceStr}</strong>
                    </div>
                    <div className="tx-plan-row">
                      <span className="tx-label">
                        Down payment ({Number(dpBps) / 100}%)
                      </span>
                      <strong>{formatEth(downPaymentPreview)}</strong>
                    </div>
                    <div className="tx-plan-row">
                      <span className="tx-label">Requested loan</span>
                      <strong>{formatEth(loanPreview)}</strong>
                    </div>
                    <p className="tx-hint tx-plan-check">
                      Down payment + requested loan = property price:{" "}
                      {formatEth(downPaymentPreview)} + {formatEth(loanPreview)}{" "}
                      = {priceStr}. The escrow locks this exact split on-chain —
                      it can never exceed or fall short of the asking price.
                    </p>
                  </div>
                  {(ready(ratePct) || ready(tenureMonths)) && (
                    <p className="tx-hint">
                      Estimated monthly EMI — informational only ≈{" "}
                      {emiEth.toLocaleString("en-IN", {
                        minimumFractionDigits: 2,
                        maximumFractionDigits: 2,
                      })}{" "}
                      test ETH (≈ ₹
                      {Math.round(emiEth * demoRateInrPerEth).toLocaleString("en-IN")}{" "}
                      / month)<br />
                      Off-chain estimate only - not a real bank quote.
                    </p>
                  )}
                  <button
                    type="button"
                    className="tx-action"
                    onClick={handleRequestFinancing}
                    disabled={
                      !buyerWorkspaceActive ||
                      !!pending ||
                      (status.inspectionRequired && !status.inspectionPassed)
                    }
                  >
                    {pending ? "Working…" : "Request financing"}
                  </button>
                </fieldset>
              )}

            {status.lenderRequired && financingRejected && (
              <p className="tx-hint">
                The lender rejected your financing request. You may adjust the
                terms above and request again, or cancel the sale.
              </p>
            )}

            {status.lenderRequired && financingApproved && (
              <div className="tx-account-actions">
                {!buyerDownComplete ? (
                  <button
                    type="button"
                    className="tx-action"
                    onClick={handlePayDownPayment}
                    disabled={!buyerWorkspaceActive || !!pending}
                  >
                    {pending
                      ? "Working…"
                      : `Pay down-payment balance (${formatEth(downPaymentWei.sub(buyerFunded))})`}
                  </button>
                ) : (
                  <span className="tx-hint">Down payment complete.</span>
                )}
              </div>
            )}

            {financingApproved && !status.buyerApproved && (
              <div className="tx-account-actions">
                <button
                  type="button"
                  className="tx-action"
                  onClick={handleApproveBuyer}
                  disabled={!buyerWorkspaceActive || !!pending}
                >
                  {pending ? "Working…" : "Approve sale (buyer)"}
                </button>
              </div>
            )}
            {(totalFunded.gt(0) || buyerFunded.gt(0)) && (
              <div className="tx-account-actions">
                <button
                  type="button"
                  className="tx-action tx-danger"
                  onClick={handleCancel}
                  disabled={!buyerWorkspaceActive || !!pending}
                >
                  {pending ? "Working…" : "Cancel & refund my test ETH"}
                </button>
              </div>
            )}
          </div>
        )}

        {listed && (
          <p className="tx-hint">
            Status: {STATUS_TEXT[statusNum] || statusNum}.{" "}
            {buyerPresent
              ? "A buyer is already committed to this property."
              : "No buyer committed yet — the first account to deposit earnest becomes the buyer."}
          </p>
        )}
      </RoleCard>

      {/* INSPECTOR */}
      <RoleCard
        title="Inspector"
        active={inspectorWorkspaceActive}
        connected={!!account}
        footer={!inspectorWorkspaceActive && performerHint("inspector", !!account)}
      >
        {status.inspectionRequired ? (
          <>
            <p className="tx-hint">
              This sale requires inspector verification before it can settle.
            </p>
            {underContract && inspectorWorkspaceActive && (
              <div className="tx-account-actions">
                <button
                  type="button"
                  className="tx-action"
                  onClick={() => handleInspect(true)}
                  disabled={!!pending || status.inspectionPassed}
                >
                  {pending ? "Working…" : "Verify property (inspection passed)"}
                </button>
                <button
                  type="button"
                  className="tx-action"
                  onClick={() => handleInspect(false)}
                  disabled={!!pending || status.inspectionPassed}
                >
                  {pending ? "Working…" : "Mark inspection failed"}
                </button>
              </div>
            )}
            <p className="tx-hint">
              {status.inspectionPassed
                ? "Inspection passed - the seller may proceed."
                : "Awaiting inspector verdict."}
            </p>
          </>
        ) : (
          <p className="tx-hint">
            Inspection: <strong>NOT REQUIRED</strong> for this sale. The seller
            listed it through the direct Model A path, so there is no inspector
            gate. This is different from a property that has not been registered
            yet.
          </p>
        )}
      </RoleCard>

      {/* LENDER */}
      <RoleCard
        title="Lender"
        active={lenderWorkspaceActive}
        connected={!!account}
        footer={!lenderWorkspaceActive && performerHint("lender", !!account)}
      >
        {status.lenderRequired ? (
          <>
            <p className="tx-hint">
              Review the buyer&#39;s financing request. On approval the lender
              disburses the loan into the escrow; the buyer pays only the down
              payment.
            </p>
            {financingRequested && finRatePct && (
              <p className="tx-hint">
                Request terms: {Number(fin.downPaymentPctBps) / 100}% down
                payment, {finRatePct}% annual interest,{" "}
                {fin.loanTenureMonths} months.
              </p>
            )}
            {!financingRequested && (
              <p className="tx-hint">No financing request from the buyer yet.</p>
            )}
            {financingRequested && !financingApproved && !financingRejected && lenderWorkspaceActive && (
              <div className="tx-account-actions">
                <button
                  type="button"
                  className="tx-action"
                  onClick={() => handleFinancingDecision(true)}
                  disabled={!!pending}
                >
                  {pending ? "Working…" : "Approve financing"}
                </button>
                <button
                  type="button"
                  className="tx-action tx-danger"
                  onClick={() => handleFinancingDecision(false)}
                  disabled={!!pending}
                >
                  {pending ? "Working…" : "Reject financing"}
                </button>
              </div>
            )}
            {financingRejected && (
              <p className="tx-hint">Financing rejected - the buyer may re-request.</p>
            )}
            {financingApproved && loanFullyFunded && (
              <p className="tx-hint">Loan fully disbursed.</p>
            )}
            {financingApproved && !loanFullyFunded && lenderWorkspaceActive && (
              <div className="tx-account-actions">
                <button
                  type="button"
                  className="tx-action"
                  onClick={handleFundLoan}
                  disabled={!!pending}
                >
                  {pending
                    ? "Working…"
                    : `Disburse loan (${formatEth(loanAmountWei.sub(lenderFunded))})`}
                </button>
                {lenderFunded.eq(0) && (
                  <button
                    type="button"
                    className="tx-action tx-danger"
                    onClick={() => handleFinancingDecision(false)}
                    disabled={!!pending}
                  >
                    {pending ? "Working…" : "Reject financing"}
                  </button>
                )}
              </div>
            )}
          </>
        ) : (
          <p className="tx-hint">
            Financing: <strong>NOT REQUIRED</strong> for this sale. The buyer is
            not requesting a loan, so there is no lender gate.
          </p>
        )}
      </RoleCard>
      </div>

      {/* Purchase Progress & Finalization */}
      <div
        className={`tx-card tx-progress-card${
          statusNum === 3 ? " tx-finalize-card--ready" : ""
        }`}
      >
        <div className="tx-card-head">
          <h4>Purchase Progress</h4>
          {statusNum === 3 && (
            <span className="tx-role-state">Ready to finalize</span>
          )}
        </div>
        {progressNote && (
          <p className="tx-hint tx-progress-note" role="status">
            {progressNote}
          </p>
        )}
        {noActiveSale ? (
          <p className="tx-hint">
            No active sale — progress appears here once the seller lists this
            property.
          </p>
        ) : (
          <ol className="tx-progress-list tx-progress-list--stages">
            {stages.map((stage) => (
              <li
                className={`tx-progress-row${
                  stage.done ? " tx-progress-row--done" : ""
                }`}
                key={stage.label}
              >
                <span className="tx-progress-mark" aria-hidden="true">
                  {stage.done ? "✓" : "·"}
                </span>
                <span className="tx-progress-label">{stage.label}</span>
                <span className="tx-progress-detail">{stage.detail}</span>
              </li>
            ))}
          </ol>
        )}

        {statusNum === 3 && (
          <div className="tx-account-actions">
            <button
              type="button"
              className="tx-action tx-action--final"
              onClick={handleFinalize}
              disabled={!finalizable || !account || !!pending}
            >
              {pending
                ? "Working…"
                : "Finalize sale - NFT to buyer, {price} to seller".replace("{price}", priceStr)}
            </button>
            {finalizable && account && (
              <p className="tx-hint">
                Any connected account may finalize the on-chain settlement; the
                escrow verifies every condition before transferring the NFT.
              </p>
            )}
            {statusNum === 3 && !finalizable && (
              <p className="tx-hint tx-warn">
                Cannot finalize yet: {finalizeBlockers.join(", ")}.
              </p>
            )}
          </div>
        )}
      </div>

      {isFinalized && (
        <div className="tx-card tx-finalized-card" role="status">
          <div className="tx-card-head">
            <h4>Transaction Finalized</h4>
            <span className="tx-chip tx-chip--sold">SOLD</span>
          </div>
          <div className="tx-plan">
            <div className="tx-plan-row">
              <span className="tx-label">Property price</span>
              <strong>
                {formatInrCompact(priceInr)} ({priceStr})
              </strong>
            </div>
            {financeMode ? (
              <>
                <div className="tx-plan-row">
                  <span className="tx-label">Down payment</span>
                  <strong>
                    {formatInrCompact(weiToInr(downPaymentWei))} (
                    {formatEth(downPaymentWei)})
                  </strong>
                </div>
                <div className="tx-plan-row">
                  <span className="tx-label">Loan</span>
                  <strong>
                    {formatInrCompact(weiToInr(loanAmountWei))} (
                    {formatEth(loanAmountWei)})
                  </strong>
                </div>
              </>
            ) : (
              <div className="tx-plan-row">
                <span className="tx-label">Full payment</span>
                <strong>{formatEth(priceWei)}</strong>
              </div>
            )}
            <div className="tx-plan-row">
              <span className="tx-label">Owner</span>
              <strong>Buyer (NFT delivered)</strong>
            </div>
          </div>
          <p className="tx-hint">
            This sale has been finalized on-chain. The property NFT was
            transferred to the buyer and the full price was paid to the seller.
          </p>
        </div>
      )}
      {isCancelled && (
        <p className="tx-hint">
          This sale has been cancelled and any funded test ETH was refunded to
          the buyer and / or the lender as applicable.
        </p>
      )}

      <details className="tx-details">
        <summary>View blockchain transaction history</summary>
        <SaleHistory
          mreidId={mreidId}
          chainRecord={chainRecord}
          refreshTick={refreshTick}
        />
      </details>

      {notice && <p className="tx-notice">{notice}</p>}
    </section>
  );
}