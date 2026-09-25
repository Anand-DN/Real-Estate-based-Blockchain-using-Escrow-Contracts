import { useCallback, useEffect, useState } from "react";
import { getSaleHistory } from "../lib/blockchain";
import { formatEth } from "../lib/format";

const TYPE_LABEL = {
  SaleListed: "Listed for sale",
  EarnestDeposited: "Earnest deposit",
  BalanceFunded: "Balance funded",
  SaleApproved: "Approved",
  SaleDisapproved: "Disapproved",
  SaleFinalized: "Finalized",
  SaleCancelled: "Cancelled",
  InspectionRequested: "Inspection requested",
  InspectionUpdated: "Inspector verification updated",
  FinancingRequested: "Financing requested by buyer",
  LoanApproved: "Loan approved",
  LoanRejected: "Loan rejected",
  BuyerDownPaymentFunded: "Buyer down payment funded",
  LoanFunded: "Lender loan funded",
  BuyerApproved: "Buyer approved",
  SellerApproved: "Seller approved",
  PropertyMinted: "NFT minted (registered)",
};

function shortAddress(value) {
  if (!value) return "—";
  const text = String(value);
  if (text.length < 12) return text;
  return `${text.slice(0, 6)}…${text.slice(-4)}`;
}

function blockExplorerLink(row) {
  return shortAddress(row.txHash);
}

export default function SaleHistory({ mreidId, chainRecord, refreshTick }) {
  const [data, setData] = useState(null);

  const load = useCallback(async () => {
    try {
      const result = await getSaleHistory(mreidId, chainRecord);
      setData(result);
    } catch {
      setData({ available: false, error: "Blockchain unavailable." });
    }
  }, [mreidId, chainRecord]);

  useEffect(() => {
    load();
  }, [load, refreshTick]);

  if (!data || !data.available || !data.registered) {
    return (
      <section className="tx-panel">
        <h3>Sale History</h3>
        <p className="tx-unavailable">
          {data && data.error
            ? data.error
            : data && data.registered === false
              ? "No blockchain activity yet — this MREID record is not tokenized on the prototype chain."
              : "No sale history available for this MREID record."}
        </p>
      </section>
    );
  }

  if (!data.rows || data.rows.length === 0) {
    return (
      <section className="tx-panel">
        <h3>Sale History</h3>
        <p className="tx-unavailable">
          This property has not been through an escrow sale yet.
        </p>
      </section>
    );
  }

  return (
    <section className="tx-panel">
      <h3>Sale History</h3>
      <p className="tx-legal">
        On-chain events for this property on chain {data.chainId || 31337} —
        prototype test chain.
      </p>
      <table className="tx-history">
        <thead>
          <tr>
            <th>Event</th>
            <th>Block</th>
            <th>Test ETH</th>
            <th>TX</th>
          </tr>
        </thead>
        <tbody>
          {data.rows.map((row) => (
            <tr key={row.key}>
              <td>{TYPE_LABEL[row.type] || row.type}</td>
              <td>{row.blockNumber}</td>
              <td>{formatEth(row.amountWei)}</td>
              <td>{blockExplorerLink(row)}</td>
            </tr>
          ))}
        </tbody>
      </table>
      <p className="tx-hint">
        Amounts shown are test ETH on the prototype chain and are not real
        money.
      </p>
    </section>
  );
}