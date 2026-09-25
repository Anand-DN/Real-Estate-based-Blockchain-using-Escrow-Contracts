import { ethers } from "ethers";

export function formatInr(value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) {
    return "—";
  }
  return `₹${Math.round(Number(value)).toLocaleString("en-IN")}`;
}

export function formatEth(value) {
  if (value === null || value === undefined) return "—";
  let wei;
  try {
    wei = ethers.BigNumber.from(value);
  } catch {
    return "—";
  }
  const eth = Number(ethers.utils.formatEther(wei));
  if (eth === 0) return "0 ETH";
  return `${eth.toLocaleString("en-IN", { maximumFractionDigits: 4 })} test ETH`;
}

// Compact Indian-format INR for dashboard stats, e.g. "₹3.00 Crore" for
// 30,000,000 and "₹60 lakh" for 6,000,000.
export function formatInrCompact(value) {
  const n = Number(value);
  if (!Number.isFinite(n)) return "—";
  const crore = 10000000;
  const lakh = 100000;
  if (Math.abs(n) >= crore) {
    return `₹${(n / crore).toLocaleString("en-IN", {
      minimumFractionDigits: 2,
      maximumFractionDigits: 2,
    })} Crore`;
  }
  if (Math.abs(n) >= lakh) {
    return `₹${(n / lakh).toLocaleString("en-IN", { maximumFractionDigits: 1 })} lakh`;
  }
  return `₹${Math.round(n).toLocaleString("en-IN")}`;
}