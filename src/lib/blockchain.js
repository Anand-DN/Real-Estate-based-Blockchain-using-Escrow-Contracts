import { ethers } from "ethers";
import config from "../config.json";
import PropertyNFT from "../abis/PropertyNFT.json";
import PropertyRegistry from "../abis/PropertyRegistry.json";
import MillowEscrow from "../abis/MillowEscrow.json";
import Escrow from "../abis/Escrow.json";
import { propertyChain } from "./millowApi";

const RPC_URL =
  process.env.REACT_APP_BLOCKCHAIN_RPC_URL || "http://127.0.0.1:8545";
const EXPECTED_CHAIN_ID = 31337;
const CHAIN_NAME = "Millow Localhost (Hardhat Network)";

const noNodeStatus = (error) => ({
  available: false,
  reason: "no-node",
  error: error ? String(error && error.message ? error.message : error) : "Blockchain unavailable.",
});

const testEnvSkipsNetwork = () =>
  process.env.NODE_ENV === "test" && !process.env.REACT_APP_BLOCKCHAIN_RPC_URL;

const errorText = (error) =>
  String(error && error.message ? error.message : error);

const chainRecord = (value) => {
  if (!value || typeof value !== "object") return null;
  if (value.chain && typeof value.chain === "object") return value.chain;
  if (value.record && typeof value.record === "object") return value.record;
  if (value.tokenized !== undefined || value.token_id !== undefined) {
    return value;
  }
  return null;
};

const getChainRecord = async (mreidId, suppliedRecord) => {
  const supplied = chainRecord(suppliedRecord);
  if (supplied) return supplied;
  try {
    return chainRecord(await propertyChain(mreidId));
  } catch {
    return null;
  }
};

const positiveTokenId = (value) => {
  try {
    const token = ethers.BigNumber.from(value);
    return token.gt(0) ? token.toNumber() : null;
  } catch {
    return null;
  }
};

const readContractValue = async (fn) => {
  try {
    return { ok: true, value: await fn() };
  } catch (error) {
    return { ok: false, value: null, error: errorText(error) };
  }
};

const resolveTokenReference = async (nft, mreidId, indexed) => {
  let direct = null;
  let directError = null;
  try {
    direct = positiveTokenId(await nft.tokenByProperty(mreidId));
  } catch (error) {
    directError = errorText(error);
  }

  const mapped = positiveTokenId(indexed && indexed.token_id);
  if (direct !== null) {
    return {
      tokenId: direct,
      mappedTokenId: mapped,
      tokenSource: "chain",
      directKnown: true,
    };
  }

  if (mapped !== null && indexed && indexed.tokenized !== false) {
    return {
      tokenId: directError ? mapped : null,
      mappedTokenId: mapped,
      tokenSource: "indexed",
      directKnown: directError === null,
      conflict: directError === null,
      error: directError,
    };
  }

  return {
    tokenId: null,
    mappedTokenId: mapped,
    tokenSource: directError ? "unknown" : "none",
    directKnown: directError === null,
    conflict: false,
    error: directError,
  };
};

const hasBytecode = async (provider, address) => {
  if (!address) return false;
  try {
    const code = await provider.getCode(address);
    return Boolean(code && code !== "0x");
  } catch {
    return false;
  }
};

export async function getPropertyChainStatus(mreidId, suppliedRecord) {
  if (testEnvSkipsNetwork()) {
    return { available: false, reason: "test", error: "Blockchain unavailable." };
  }

  let provider;
  try {
    provider = new ethers.providers.JsonRpcProvider(RPC_URL);
  } catch (err) {
    return noNodeStatus(err);
  }

  let network;
  try {
    network = await provider.getNetwork();
  } catch (err) {
    return noNodeStatus(err);
  }

  const chainId = network.chainId;
  const supported = chainId === EXPECTED_CHAIN_ID;
  const chainName = supported ? CHAIN_NAME : network.name || `Chain ${chainId}`;
  const chainConfig = config[String(chainId)];
  const hasContracts =
    chainConfig && chainConfig.propertyNft && chainConfig.propertyRegistry;

  if (!hasContracts) {
    return {
      available: true,
      chainId,
      chainName,
      supported: false,
      registered: false,
      reason: "no-contracts",
      error: `No MILLOW contracts configured for chain ${chainId}. Run scripts/deployMillow.js.`,
    };
  }

  const nftAddress = chainConfig.propertyNft.address;
  const registryAddress = chainConfig.propertyRegistry.address;
  const nft = new ethers.Contract(nftAddress, PropertyNFT, provider);
  const registry = new ethers.Contract(
    registryAddress,
    PropertyRegistry,
    provider,
  );
  const indexed = await getChainRecord(mreidId, suppliedRecord);
  const base = {
    available: true,
    chainId,
    chainName,
    supported,
    nftAddress,
    registryAddress,
    mreidId,
    indexedChain: indexed,
  };

  const tokenReference = await resolveTokenReference(nft, mreidId, indexed);
  if (tokenReference.conflict) {
    return {
      ...base,
      registered: null,
      tokenizationUnknown: true,
      reason: "indexed-chain-mismatch",
      error: "The indexed token ID conflicts with the connected contract.",
    };
  }

  if (tokenReference.tokenId === null) {
    if (tokenReference.directKnown) {
      return {
        ...base,
        registered: false,
        reason: "not-tokenized",
      };
    }
    return {
      ...base,
      registered: null,
      tokenizationUnknown: true,
      reason: "mapping-unavailable",
      error: tokenReference.error || "The token mapping could not be verified.",
    };
  }

  const id = tokenReference.tokenId;
  const escrowAddress = chainConfig.millowEscrow
    ? chainConfig.millowEscrow.address
    : null;
  const escrowDeployed = await hasBytecode(provider, escrowAddress);
  const escrow = escrowDeployed
    ? new ethers.Contract(escrowAddress, MillowEscrow, provider)
    : null;

  const [ownerRead, tokenURIRead, propertyRead] = await Promise.all([
    readContractValue(() => nft.ownerOf(id)),
    readContractValue(() => nft.tokenURI(id)),
    readContractValue(() => nft.propertyOf(id)),
  ]);

  if (!ownerRead.ok) {
    return {
      ...base,
      registered: null,
      tokenizationUnknown: true,
      reason: "token-unavailable",
      error: ownerRead.error,
      tokenId: id,
      tokenSource: tokenReference.tokenSource,
    };
  }

  if (
    propertyRead.ok &&
    propertyRead.value &&
    String(propertyRead.value).toLowerCase() !== String(mreidId).toLowerCase()
  ) {
    return {
      ...base,
      registered: null,
      tokenizationUnknown: true,
      reason: "property-mismatch",
      error: "The connected token points to a different MREID.",
      tokenId: id,
      tokenSource: tokenReference.tokenSource,
    };
  }

  let listingRead = await readContractValue(() => registry.isOnOffer(id));
  let listingSource = "registry";
  if (!listingRead.ok) {
    const legacyRegistry = new ethers.Contract(registryAddress, Escrow, provider);
    const legacyListing = await readContractValue(() =>
      legacyRegistry.isListed(id),
    );
    if (legacyListing.ok) {
      listingRead = legacyListing;
      listingSource = "legacy-registry";
    }
  }

  const escrowOfferRead = escrow
    ? await readContractValue(() => escrow.isOnOffer(id))
    : { ok: false, value: null, error: "MILLOW escrow is not deployed." };
  const saleRead = escrow
    ? await readContractValue(() => escrow.sales(id))
    : { ok: false, value: null, error: "MILLOW escrow is not deployed." };

  let seller = null;
  if (listingRead.ok && Boolean(listingRead.value)) {
    const sellerRead =
      listingSource === "legacy-registry"
        ? await readContractValue(() =>
            new ethers.Contract(registryAddress, Escrow, provider).sellerOf(id),
          )
        : await readContractValue(() => registry.sellerOf(id));
    seller = sellerRead.ok ? sellerRead.value : null;
  }

  const knownOfferValues = [];
  if (listingRead.ok) knownOfferValues.push(Boolean(listingRead.value));
  if (escrowOfferRead.ok) knownOfferValues.push(Boolean(escrowOfferRead.value));
  const onOffer = knownOfferValues.length ? knownOfferValues.some(Boolean) : null;
  const saleStatus = saleRead.ok ? Number(saleRead.value.status) : null;
  const finalized = saleStatus === null ? null : saleStatus === 4;

  let mintTxHash = null;
  let mintBlock = null;
  try {
    const filter = nft.filters.PropertyMinted(id);
    const logs = await provider.getLogs({
      ...filter,
      fromBlock: 0,
      toBlock: "latest",
    });
    const last = logs[logs.length - 1];
    if (last) {
      mintTxHash = last.transactionHash;
      mintBlock = last.blockNumber;
    }
  } catch (err) {
    mintTxHash = null;
    mintBlock = null;
  }

  return {
    ...base,
    registered: true,
    tokenId: id,
    tokenSource: tokenReference.tokenSource,
    owner: ownerRead.value,
    propertyId: propertyRead.ok ? propertyRead.value : null,
    tokenURI: tokenURIRead.ok ? tokenURIRead.value : null,
    onOffer,
    listingSource,
    listingKnown: knownOfferValues.length > 0,
    workflowAvailable: escrowDeployed,
    chainVariant: escrowDeployed ? "millow" : "legacy",
    seller,
    saleStatus,
    finalized,
    mintTxHash,
    mintBlock,
  };
}

const SALE_EVENTS = [
  "SaleListed",
  "EarnestDeposited",
  "BalanceFunded",
  "SaleApproved",
  "SaleDisapproved",
  "SaleFinalized",
  "SaleCancelled",
  "InspectionRequested",
  "InspectionUpdated",
  "FinancingRequested",
  "LoanApproved",
  "LoanRejected",
  "BuyerDownPaymentFunded",
  "LoanFunded",
  "BuyerApproved",
  "SellerApproved",
];

const STATUS_NAMES = {
  0: "Not in a sale",
  1: "Listed",
  2: "Under contract",
  3: "Approved",
  4: "Finalized",
  5: "Cancelled",
};

// AccessControl role ids as the escrow derives them.
const INSPECTOR_ROLE = ethers.utils.id("INSPECTOR_ROLE");
const LENDER_ROLE = ethers.utils.id("LENDER_ROLE");

export const ROLE_NAMES = {
  seller: "Seller",
  buyer: "Buyer",
  inspector: "Inspector",
  lender: "Lender",
  viewer: "Viewer",
};

// Canonical demo identities from config.json.  demoRoleOf is used for
// identity-badge fallbacks (e.g. unregistered properties and the commit
// candidate gate); it must NOT override on-chain facts such as real NFT
// ownership or who is committed on an escrow sale.
export function demoRoleOf(account) {
  if (!account) return null;
  const acct = String(account).toLowerCase();
  const roles = (config && config.demoRoles) || {};
  const order = ["seller", "buyer", "inspector", "lender"];
  for (const key of order) {
    const addr = roles[key];
    if (addr && String(addr).toLowerCase() === acct) return key;
  }
  return null;
}

// Who is the connected account on this sale?  On-chain facts always win —
// the demo identity badge must never override actual ownership or commitment:
//   1. the escrow's committed buyer,
//   2. the escrow's sale seller,
//   3. the actual NFT owner (the only account that may act as seller),
//   4. an assigned Inspector / Lender role on the escrow,
//   5. viewer.
// resolveAccountRole is fed through getSaleStatus so every caller sees the
// same authoritative role for the connected account on a registered property.
export async function resolveAccountRole(escrow, sale, account, ownerOf) {
  if (!account) return "viewer";
  const acct = account.toLowerCase();
  const zero = ethers.constants.AddressZero.toLowerCase();
  const buyer = sale.buyer ? sale.buyer.toLowerCase() : null;
  const seller = sale.seller ? sale.seller.toLowerCase() : null;
  if (buyer && buyer !== zero && buyer === acct) return "buyer";
  if (seller && seller !== zero && seller === acct) return "seller";
  if (ownerOf && String(ownerOf).toLowerCase() === acct) return "seller";
  const [inspector, lender] = await Promise.all([
    escrow.hasRole(INSPECTOR_ROLE, account),
    escrow.hasRole(LENDER_ROLE, account),
  ]);
  if (inspector) return "inspector";
  if (lender) return "lender";
  return "viewer";
}

const saleSetup = async (mreidId, suppliedRecord) => {
  const status = await getPropertyChainStatus(mreidId, suppliedRecord);
  if (!status || !status.available || !status.registered || !status.supported) {
    return status;
  }

  const chainConfig = config[String(status.chainId)];
  const escrowAddress = chainConfig.millowEscrow
    ? chainConfig.millowEscrow.address
    : null;
  const provider = new ethers.providers.JsonRpcProvider(RPC_URL);
  const nft = new ethers.Contract(status.nftAddress, PropertyNFT, provider);
  const escrow = status.workflowAvailable
    ? new ethers.Contract(escrowAddress, MillowEscrow, provider)
    : null;

  return {
    ...status,
    escrowAddress,
    provider,
    nft,
    escrow,
  };
};

export async function getSaleStatus(mreidId, account, suppliedRecord) {
  const setup = await saleSetup(mreidId, suppliedRecord);
  if (!setup.registered || !setup.available) return setup;
  if (!setup.workflowAvailable || !setup.escrow) {
    return {
      ...setup,
      reason: "workflow-unavailable",
      error: "The NFT is visible, but the MILLOW escrow contract is not deployed on this chain.",
    };
  }

  try {
    const [sale, onOffer, ownerOf] = await Promise.all([
      setup.escrow.sales(setup.tokenId),
      setup.escrow.isOnOffer(setup.tokenId),
      setup.nft.ownerOf(setup.tokenId),
    ]);
    const status = Number(sale.status);
    const financing = {
      lender: sale.financing.lender,
      requested: sale.financing.requested,
      approved: sale.financing.approved,
      rejected: sale.financing.rejected,
      downPaymentPctBps: Number(sale.financing.downPaymentPctBps),
      interestRateBps: sale.financing.interestRateBps,
      loanTenureMonths: Number(sale.financing.loanTenureMonths),
      downPaymentWei: sale.financing.downPaymentWei,
      loanAmountWei: sale.financing.loanAmountWei,
    };
    let accountRole = "viewer";
    if (account) {
      accountRole = await resolveAccountRole(
        setup.escrow,
        sale,
        account,
        ownerOf,
      );
    }
    return {
      ...setup,
      onOffer,
      status,
      statusName: STATUS_NAMES[status] || String(status),
      ownerOf,
      seller: sale.seller,
      buyer: sale.buyer,
      priceWei: sale.priceWei,
      earnestWei: sale.earnestWei,
      buyerFundedWei: sale.buyerFundedWei,
      lenderFundedWei: sale.lenderFundedWei,
      fundedWei: sale.buyerFundedWei.add(sale.lenderFundedWei),
      sellerApproved: sale.sellerApproved,
      buyerApproved: sale.buyerApproved,
      inspectionPassed: sale.inspectionPassed,
      inspectionRequired: sale.inspectionRequired,
      lenderRequired: sale.lenderRequired,
      financing,
      accountRole,
    };
  } catch (err) {
    return {
      ...setup,
      reason: "error",
      error: String(err && err.message ? err.message : err),
    };
  }
}

export async function getSaleHistory(mreidId, suppliedRecord) {
  const setup = await saleSetup(mreidId, suppliedRecord);
  if (!setup.registered || !setup.available) return setup;
  if (!setup.workflowAvailable || !setup.escrow) {
    return { ...setup, rows: [], historyAvailable: false };
  }

  try {
    const iface = new ethers.utils.Interface(MillowEscrow);
    const filter = {
      address: setup.escrowAddress,
      fromBlock: 0,
      toBlock: "latest",
    };
    const logs = await setup.provider.getLogs(filter);
    const rows = [];
    for (const log of logs) {
      let parsed;
      try {
        parsed = iface.parseLog(log);
      } catch {
        continue;
      }
      if (!SALE_EVENTS.includes(parsed.name)) continue;
      if (parsed.args.tokenId && parsed.args.tokenId.toString() !== String(setup.tokenId)) {
        continue;
      }
      rows.push({
        key: `${log.blockNumber}:${log.transactionIndex}:${parsed.name}`,
        type: parsed.name,
        txHash: log.transactionHash,
        blockNumber: log.blockNumber,
        amountWei: parsed.args.amountWei || parsed.args.priceWei || parsed.args.refundWei || null,
        actor: parsed.args.buyer || parsed.args.seller || parsed.args.approver || parsed.args.recipient || parsed.args.lender || null,
      });
    }

    // A registered property was minted once as an NFT; surface that on-chain
    // event too so the history explains how the token came to exist.
    try {
      const nftIface = new ethers.utils.Interface(PropertyNFT);
      const mintFilter = setup.nft.filters.PropertyMinted(setup.tokenId);
      const mintLogs = await setup.provider.getLogs({
        ...mintFilter,
        fromBlock: 0,
        toBlock: "latest",
      });
      for (const log of mintLogs) {
        let parsed;
        try {
          parsed = nftIface.parseLog(log);
        } catch {
          continue;
        }
        // PropertyMinted has an indexed string propertyId, so ethers v5
        // parseLog returns only positional args here (named args are dropped).
        if (parsed.args[0].toString() !== String(setup.tokenId)) continue;
        rows.push({
          key: `${log.blockNumber}:${log.transactionIndex}:PropertyMinted`,
          type: "PropertyMinted",
          txHash: log.transactionHash,
          blockNumber: log.blockNumber,
          amountWei: null,
          actor: parsed.args[2] || null,
        });
      }
    } catch {
      // Mint log lookup is best-effort; escrow events alone are still valid.
    }

    rows.sort((a, b) => b.blockNumber - a.blockNumber || b.key.localeCompare(a.key));
    return { ...setup, rows };
  } catch (err) {
    return {
      ...setup,
      reason: "error",
      error: String(err && err.message ? err.message : err),
    };
  }
}