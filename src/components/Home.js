import { ethers } from 'ethers';
import { useEffect, useState, useCallback } from 'react';

import close from '../assets/close.svg';
import Contact from './Contact';

const Home = ({ home, provider, account, escrow, togglePop, setNotification }) => {
    // Metadata IDs are not guaranteed to be the NFT ID for user-created
    // listings.  The app attaches tokenId while loading the collection.
    const propertyId = home.tokenId ?? home.id
    const [hasBought, setHasBought] = useState(false)
    const [hasLended, setHasLended] = useState(false)
    const [hasInspected, setHasInspected] = useState(false)
    const [hasSold, setHasSold] = useState(false)

    const [buyer, setBuyer] = useState(null)
    const [lender, setLender] = useState(null)
    const [inspector, setInspector] = useState(null)
    const [seller, setSeller] = useState(null)

    const [owner, setOwner] = useState(null)
    const [showContact, setShowContact] = useState(false)
    const [txPending, setTxPending] = useState(false)

    const [insights, setInsights] = useState(null)
    const [aiStatus, setAiStatus] = useState('loading')

    const AI_BASE = process.env.REACT_APP_AI_URL || 'http://localhost:8000'

    const withTimeout = (promise, ms = 60000) => {
        return Promise.race([
            promise,
            new Promise((_, reject) => {
                setTimeout(() => {
                    reject(new Error('Transaction was not mined within the time limit'))
                }, ms)
            }),
        ])
    }

    const waitForSuccess = async (transaction) => {
        // A timeout keeps `txPending` (and the disabled buttons) from being
        // stuck forever when MetaMask leaves a transaction queued/pending.
        const receipt = await withTimeout(transaction.wait())
        // A transaction is successful only when MetaMask/RPC returns a mined
        // receipt with status 1. Never show an approval for an unmined or
        // reverted transaction.
        if (!receipt || receipt.status !== 1) {
            throw new Error('The transaction was reverted')
        }
        return receipt
    }

    const getTransactionError = (error) => {
        console.error('Millow transaction error:', error)
        if (error?.code === 4001) return 'Transaction rejected in MetaMask'

        const message = error?.reason || error?.data?.message || error?.error?.message || error?.message || ''

        // Give clear guidance for the common local-node failures instead of a
        // cryptic MetaMask error.
        if (/nonce too (low|high)|replacement transaction underpriced|already known|inconsistent nonce/i.test(message)) {
            return 'MetaMask nonce is out of sync with the local node. Open MetaMask → Settings → Advanced → Clear activity tab data, restart the Hardhat node, then retry.'
        }
        if (/only buyer can call this method/i.test(message)) {
            return 'This property is already committed to another buyer.'
        }

        return message || 'Transaction failed'
    }

    const getWalletSigner = async () => {
        if (!provider || !window.ethereum) throw new Error('Wallet is not connected')
        const accounts = await window.ethereum.request({ method: 'eth_requestAccounts' })
        const selected = accounts?.[0]
        if (!selected) throw new Error('Wallet is not connected')
        // Always sign with the account MetaMask has selected right now, not a
        // possibly stale copy in React state. Signing from an account MetaMask
        // does not control fails as "unauthorized" for every transaction.
        return provider.getSigner(selected)
    }

    const requireRoleSigner = async (signer, expected, role) => {
        const actual = (await signer.getAddress()).toLowerCase()
        if (!expected || actual !== expected.toLowerCase()) {
            throw new Error(`${role} wallet required. Connected wallet is ${actual.slice(0, 10)}...`)
        }
    }

    const toggleContact = () => {
        showContact ? setShowContact(false) : setShowContact(true)
    }

    const fetchDetails = useCallback(async () => {
        // -- Buyer

        const buyer = await escrow.buyer(propertyId)
        setBuyer(buyer)

        // Buyer's step is complete only after depositing earnest AND approving.
        // The raw `approval` flag alone must not disable the Buy button: it can
        // become true after a partial/failed flow.
        const isCommittedBuyer = Boolean(
            buyer &&
            buyer !== ethers.constants.AddressZero &&
            account &&
            buyer.toLowerCase() === account.toLowerCase()
        )
        const hasBought = isCommittedBuyer && (await escrow.approval(propertyId, buyer))
        setHasBought(hasBought)

        // -- Seller

        const seller = await escrow.sellerOf(propertyId)
        setSeller(seller)

        // A sale is done for the seller only when the listing is gone
        // (finalized or cancelled). The seller's `approval` is just one step
        // and must not disable the Sell button while the sale is still open.
        const isListed = await escrow.isListed(propertyId)
        setHasSold(!isListed)

        // -- Lender

        const lender = await escrow.lender()
        setLender(lender)

        const hasLenderApproval = await escrow.approval(propertyId, lender)
        const [currentBalance, purchasePrice] = await Promise.all([
            escrow.getBalance(),
            escrow.purchasePrice(propertyId),
        ])
        // Approval and funding are separate transactions.  Do not disable the
        // lender button when only approval succeeded; funding may still need a
        // retry.
        setHasLended(hasLenderApproval && currentBalance.gte(purchasePrice))

        // -- Inspector

        const inspector = await escrow.inspector()
        setInspector(inspector)

        const hasInspected = await escrow.inspectionPassed(propertyId)
        setHasInspected(hasInspected)
    }, [escrow, propertyId, account])

    const fetchOwner = useCallback(async () => {
        if (await escrow.isListed(propertyId)) return

        const owner = await escrow.buyer(propertyId)
        setOwner(owner)
    }, [escrow, propertyId])

    useEffect(() => {
        let cancelled = false
        setAiStatus('loading')
        setInsights(null)
        fetch(`${AI_BASE}/property/${propertyId}/insights`)
            .then((res) => {
                if (!res.ok) throw new Error('AI server unavailable')
                return res.json()
            })
            .then((data) => {
                if (!cancelled) {
                    setInsights(data)
                    setAiStatus('ready')
                }
            })
            .catch(() => {
                if (!cancelled) setAiStatus('error')
            })
        return () => {
            cancelled = true
        }
    }, [propertyId, AI_BASE])

    const buyHandler = async () => {
        setTxPending(true)
        try {
            const signer = await getWalletSigner()
            const signerAddress = (await signer.getAddress()).toLowerCase()

            // Never try to commit if another account already deposited earnest:
            // the contract would revert with "Only buyer can call this method"
            // and MetaMask would show it as a failed transaction.
            const currentBuyer = await escrow.buyer(propertyId)
            const hasBuyer = currentBuyer !== ethers.constants.AddressZero
            if (hasBuyer && currentBuyer.toLowerCase() !== signerAddress) {
                throw new Error('This property is already committed to another buyer')
            }

            // Deposit earnest only if this account is not already committed as
            // the buyer. Re-running the deposit would send a second earnest.
            if (!hasBuyer) {
                const escrowAmount = await escrow.escrowAmount(propertyId)
                const transaction = await escrow.connect(signer).depositEarnest(propertyId, { value: escrowAmount })
                await waitForSuccess(transaction)
            }

            // Approve the sale only if not already approved.
            if (!(await escrow.approval(propertyId, signerAddress))) {
                const transaction = await escrow.connect(signer).approveSale(propertyId)
                await waitForSuccess(transaction)
            }

            setHasBought(true)
            setNotification('Purchase approved', 'success')
            window.dispatchEvent(new Event('millow:dashboard-refresh'))
        } catch (error) {
            setNotification(getTransactionError(error), 'error')
            // Re-sync with the chain so a partially completed flow (e.g.
            // earnest deposited but approval failed) is reflected in the UI.
            fetchDetails().catch(() => {})
            fetchOwner().catch(() => {})
        } finally {
            setTxPending(false)
        }
    }

    const inspectHandler = async () => {
        setTxPending(true)
        try {
            const signer = await getWalletSigner()
            await requireRoleSigner(signer, inspector, 'Inspector')

            // Inspector updates status
            const transaction = await escrow.connect(signer).updateInspectionStatus(propertyId, true)
            await waitForSuccess(transaction)

            setHasInspected(true)
            setNotification('Inspection approved', 'success')
        } catch (error) {
            setNotification(getTransactionError(error), 'error')
        } finally {
            setTxPending(false)
        }
    }

    const cancelHandler = async () => {
        setTxPending(true)
        try {
            const signer = await getWalletSigner()
            const signerAddress = (await signer.getAddress()).toLowerCase()

            if (signerAddress === inspector?.toLowerCase()) {
                let transaction = await escrow.connect(signer).updateInspectionStatus(propertyId, false)
                await waitForSuccess(transaction)
            } else if (signerAddress === lender?.toLowerCase()) {
                let transaction = await escrow.connect(signer).disapproveSale(propertyId)
                await waitForSuccess(transaction)
            }

            const transaction = await escrow.connect(signer).cancelSale(propertyId)
            await waitForSuccess(transaction)

            if (signerAddress === inspector?.toLowerCase()) {
                setNotification('Inspection disapproved', 'error')
            } else if (signerAddress === lender?.toLowerCase()) {
                setNotification('Loan disapproved', 'error')
            } else {
                setNotification('Sale cancelled', 'error')
            }

            togglePop()
        } catch (error) {
            setNotification(getTransactionError(error), 'error')
            fetchDetails().catch(() => {})
            fetchOwner().catch(() => {})
        } finally {
            setTxPending(false)
        }
    }

    const lendHandler = async () => {
        setTxPending(true)
        try {
            const signer = await getWalletSigner()
            await requireRoleSigner(signer, lender, 'Lender')

            // Lender approves...
            const alreadyApproved = await escrow.approval(propertyId, lender)
            if (!alreadyApproved) {
                const transaction = await escrow.connect(signer).approveSale(propertyId)
                await waitForSuccess(transaction)
            }

            // Lender sends funds to contract...
            const lendAmount = (await escrow.purchasePrice(propertyId)).sub(await escrow.escrowAmount(propertyId))
            const fundingTransaction = await signer.sendTransaction({ to: escrow.address, value: lendAmount })
            await waitForSuccess(fundingTransaction)

            setHasLended(true)
            setNotification('Loan approved', 'success')
        } catch (error) {
            setNotification(getTransactionError(error), 'error')
        } finally {
            setTxPending(false)
        }
    }

    const sellHandler = async () => {
        setTxPending(true)
        try {
            const signer = await getWalletSigner()
            await requireRoleSigner(signer, seller, 'Seller')

            // Check all prerequisites BEFORE sending any transaction. The old
            // code approved first and only then told you it was waiting, so the
            // button needed a second click to finalize. Now nothing is sent
            // until the sale can actually go through.
            const [inspectionOk, buyerApproved, sellerApproved, lenderApproved, balance, price] = await Promise.all([
                escrow.inspectionPassed(propertyId),
                escrow.approval(propertyId, buyer),
                escrow.approval(propertyId, seller),
                escrow.approval(propertyId, lender),
                escrow.getBalance(),
                escrow.purchasePrice(propertyId),
            ])

            if (!inspectionOk) {
                setNotification('Waiting for inspection to pass', 'error')
                return
            }
            if (!buyerApproved) {
                setNotification('Waiting for buyer approval', 'error')
                return
            }
            if (!lenderApproved) {
                setNotification('Waiting for lender approval', 'error')
                return
            }
            if (balance.lt(price)) {
                setNotification('Waiting for full funds in escrow', 'error')
                return
            }

            // Approve only if this seller has not already approved, so the sale
            // finalizes in one click when everything is ready.
            if (!sellerApproved) {
                const approveTx = await escrow.connect(signer).approveSale(propertyId)
                await waitForSuccess(approveTx)
            }

            // All conditions met — finalize the sale
            const finalizeTx = await escrow.connect(signer).finalizeSale(propertyId)
            await waitForSuccess(finalizeTx)

            setHasSold(true)
            setNotification('Sale finalized!', 'success')
            window.dispatchEvent(new Event('millow:dashboard-refresh'))
        } catch (error) {
            setNotification(getTransactionError(error), 'error')
            // Re-sync with the chain so a partially completed flow (e.g. the
            // approval mined but finalization reverted) is reflected in the UI.
            fetchDetails().catch(() => {})
            fetchOwner().catch(() => {})
        } finally {
            setTxPending(false)
        }
    }

    useEffect(() => {
        fetchDetails()
        fetchOwner()
    }, [hasSold, account, fetchDetails, fetchOwner])

    return (
        <div className="home">
            <div className='home__details'>
                <div className="home__image">
                    <img src={home.image} alt="Home" />
                </div>
                <div className="home__overview">
                    <h1>{home.name}</h1>
                    <p>
                        <strong>{home.attributes[2].value}</strong> bds |
                        <strong>{home.attributes[3].value}</strong> ba |
                        <strong>{home.attributes[4].value}</strong> sqft
                    </p>
                    <p>{home.address}</p>

                    <h2>{home.attributes[0].value} ETH</h2>

                    {owner ? (
                        <div className='home__owned'>
                            Owned by {owner.slice(0, 6) + '...' + owner.slice(38, 42)}
                        </div>
                    ) : (
                        <div>
                            {(account === inspector) ? (
                                <div className='home__actions'>
                                    <button className='home__buy' onClick={inspectHandler} disabled={hasInspected || txPending}>
                                        Approve Inspection
                                    </button>
                                    <button className='home__cancel' onClick={cancelHandler} disabled={txPending}>Cancel Sale</button>
                                </div>
                            ) : (account === lender) ? (
                                <div className='home__actions'>
                                    <button className='home__buy' onClick={lendHandler} disabled={hasLended || txPending}>
                                        Approve & Lend
                                    </button>
                                    <button className='home__cancel' onClick={cancelHandler} disabled={txPending}>Cancel Sale</button>
                                </div>
                            ) : (account === seller) ? (
                                <div className='home__actions'>
                                    <button className='home__buy' onClick={sellHandler} disabled={hasSold || txPending}>
                                        Approve & Sell
                                    </button>
                                    <button className='home__cancel' onClick={cancelHandler} disabled={txPending}>Cancel Sale</button>
                                </div>
                            ) : (buyer && buyer !== ethers.constants.AddressZero && buyer.toLowerCase() !== account.toLowerCase()) ? (
                                <div className="home__taken">
                                    Already under contract — committed to another buyer
                                </div>
                            ) : (
                                <button className='home__buy' onClick={buyHandler} disabled={hasBought || txPending}>
                                    {hasBought ? 'Purchase approved' : 'Buy'}
                                </button>
                            )}

                            <button className='home__contact' onClick={toggleContact} disabled={txPending}>
                                Contact agent
                            </button>
                        </div>
                    )}

                    <hr />

                    <h2>Overview</h2>

                    <p>
                        {home.description}
                    </p>

                    <hr />

                    <h2>Facts and features</h2>

                    <ul>
                        {home.attributes.map((attribute, index) => (
                            <li key={index}><strong>{attribute.trait_type}</strong> : {attribute.value}</li>
                        ))}
                    </ul>

                    <hr />

                    <h2>AI Insights</h2>

                    {aiStatus === 'loading' && (
                        <p className="ai__note">Loading AI valuation...</p>
                    )}

                    {aiStatus === 'error' && (
                        <p className="ai__note">
                            AI insights unavailable. Start the AI server with <code>npm run ai</code> and refresh.
                        </p>
                    )}

                    {aiStatus === 'ready' && insights && (
                        <div className="ai__panel">
                            <div className="ai__row">
                                <span>Predicted value</span>
                                <strong>{insights.predicted_price_eth} ETH</strong>
                            </div>
                            <div className="ai__row">
                                <span>Listed price</span>
                                <strong>{insights.listed_price_eth} ETH</strong>
                            </div>
                            <div className="ai__row">
                                <span>Difference</span>
                                <span className={`ai__verdict ai__verdict--${insights.verdict.toLowerCase().replace(' ', '-')}`}>
                                    {insights.difference_pct >= 0 ? '+' : ''}{insights.difference_pct}% · {insights.verdict}
                                </span>
                            </div>
                            <div className="ai__row">
                                <span>Transaction risk</span>
                                <span className={`ai__risk ai__risk--${insights.risk.level.toLowerCase()}`}>
                                    {insights.risk.level} · {insights.risk.overall}/100
                                </span>
                            </div>
                            <div className="ai__row">
                                <span>Anomaly score</span>
                                <span className={insights.anomaly_score > 0.55 ? 'ai__bad' : 'ai__good'}>
                                    {insights.anomaly_score}{insights.anomaly_score > 0.55 ? ' · Suspicious' : ' · Normal'}
                                </span>
                            </div>

                            <h3 className="ai__subtitle">Risk breakdown</h3>
                            <ul className="ai__breakdown">
                                {Object.entries(insights.risk.breakdown).map(([key, value]) => (
                                    <li key={key}>{key}: <strong>{value}</strong></li>
                                ))}
                            </ul>

                            <h3 className="ai__subtitle">Similar properties</h3>
                            <ul className="ai__similar">
                                {insights.similar.map((similar) => (
                                    <li key={similar.token_id}>
                                        <img src={similar.image} alt={similar.name} />
                                        <div>
                                            <strong>{similar.name}</strong>
                                            <span>{similar.price_eth} ETH</span>
                                        </div>
                                        <span className="ai__match">{similar.match}% match</span>
                                    </li>
                                ))}
                            </ul>
                        </div>
                    )}
                </div>


                <button onClick={togglePop} className="home__close" disabled={txPending}>
                    <img src={close} alt="Close" />
                </button>
            </div>

            {txPending && (
                <div className="home__loading">
                    <div className="spinner"></div>
                    <p>Waiting for confirmation...</p>
                </div>
            )}

            {showContact && (
                <Contact seller={seller} toggleContact={toggleContact} setNotification={setNotification} />
            )}
        </div >
    );
}

export default Home;
