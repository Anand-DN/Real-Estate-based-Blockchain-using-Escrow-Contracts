export const hasWallet = () =>
  typeof window !== "undefined" && !!(window.ethereum && window.ethereum.request);

export async function connectWallet() {
  if (!hasWallet()) {
    return { ok: false, error: "No Web3 wallet detected." };
  }
  try {
    const accounts = await window.ethereum.request({
      method: "eth_requestAccounts",
    });
    if (!accounts || !accounts.length) {
      return { ok: false, error: "No account returned by the wallet." };
    }
    return { ok: true, account: accounts[0] };
  } catch (err) {
    return {
      ok: false,
      error: err && err.message ? err.message : "Wallet connection failed.",
    };
  }
}

export async function getSigner() {
  if (!hasWallet()) return null;
  try {
    const { ethers } = await import("ethers");
    await window.ethereum.request({ method: "eth_requestAccounts" });
    const provider = new ethers.providers.Web3Provider(window.ethereum);
    const signer = provider.getSigner();
    return signer;
  } catch {
    return null;
  }
}

// Live wallet events: the dashboard re-syncs when the user switches the active
// MetaMask account or the connected chain.  Returns an unsubscribe function.
export function subscribeWallet({ onAccount, onNetwork } = {}) {
  if (!hasWallet()) return () => {};
  const accountsHandler = (accounts) => {
    if (onAccount) onAccount(accounts && accounts.length ? accounts[0] : null);
  };
  const networkHandler = () => {
    if (onNetwork) onNetwork();
  };
  if (window.ethereum && window.ethereum.on) {
    window.ethereum.on("accountsChanged", accountsHandler);
    window.ethereum.on("chainChanged", networkHandler);
  }
  return () => {
    if (window.ethereum && window.ethereum.removeListener) {
      window.ethereum.removeListener("accountsChanged", accountsHandler);
      window.ethereum.removeListener("chainChanged", networkHandler);
    }
  };
}