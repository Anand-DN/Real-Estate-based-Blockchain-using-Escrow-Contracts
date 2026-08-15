import { useState } from 'react';
import { ethers } from 'ethers';

import close from '../assets/close.svg';

const List = ({ provider, account, escrow, realEstate, toggleList, setNotification }) => {
    const [name, setName] = useState('')
    const [address, setAddress] = useState('')
    const [description, setDescription] = useState('')
    const [beds, setBeds] = useState('')
    const [baths, setBaths] = useState('')
    const [sqft, setSqft] = useState('')
    const [year, setYear] = useState('')
    const [price, setPrice] = useState('')
    const [escrowAmount, setEscrowAmount] = useState('')
    const [sellerAddress, setSellerAddress] = useState('')
    const [image, setImage] = useState(null)
    const [txPending, setTxPending] = useState(false)

    const sellerVerified = Boolean(
        sellerAddress &&
        account &&
        sellerAddress.toLowerCase() === account.toLowerCase()
    )

    const onImageChange = (e) => {
        const file = e.target.files[0]
        if (!file) return

        const reader = new FileReader()
        reader.onload = () => setImage(reader.result)
        reader.readAsDataURL(file)
    }

    const submitHandler = async (e) => {
        e.preventDefault()
        setTxPending(true)
        try {
            const signer = provider.getSigner()

            const response = await fetch('http://localhost:3001/api/list', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    name,
                    address,
                    description,
                    image,
                    attributes: [
                        { trait_type: 'Purchase Price', value: Number(price) },
                        { trait_type: 'Type of Residence', value: 'Single family residence' },
                        { trait_type: 'Bed Rooms', value: Number(beds) },
                        { trait_type: 'Bathrooms', value: Number(baths) },
                        { trait_type: 'Square Feet', value: Number(sqft) },
                        { trait_type: 'Year Built', value: Number(year) },
                    ],
                }),
            })

            if (!response.ok) throw new Error('Failed to save property metadata')
            const { uri } = await response.json()

            // Mint the property NFT
            let transaction = await realEstate.connect(signer).mint(uri)
            await transaction.wait()

            // Use the on-chain token id (chain counter, not the metadata file id)
            const tokenId = await realEstate.totalSupply()

            // Approve escrow to transfer the NFT
            transaction = await realEstate.connect(signer).approve(escrow.address, tokenId)
            await transaction.wait()

            // List on escrow
            transaction = await escrow.connect(signer).list(
                tokenId,
                ethers.utils.parseUnits(price, 'ether'),
                ethers.utils.parseUnits(escrowAmount, 'ether')
            )
            await transaction.wait()

            setNotification('Property listed', 'success')
            toggleList()
            setTimeout(() => window.location.reload(), 1500)
        } catch (error) {
            console.error(error)
            setNotification('Listing failed', 'error')
        } finally {
            setTxPending(false)
        }
    }

    return (
        <div className="list">
            <div className="list__details">
                <h2>List your home</h2>

                <form className="list__form" onSubmit={submitHandler}>
                    <input type="text" placeholder="Property name" value={name} onChange={(e) => setName(e.target.value)} required disabled={txPending} />
                    <input type="text" placeholder="Address" value={address} onChange={(e) => setAddress(e.target.value)} required disabled={txPending} />
                    <textarea placeholder="Description" value={description} onChange={(e) => setDescription(e.target.value)} disabled={txPending}></textarea>

                    <div className="list__row">
                        <input type="number" placeholder="Bedrooms" value={beds} onChange={(e) => setBeds(e.target.value)} required disabled={txPending} />
                        <input type="number" placeholder="Bathrooms" value={baths} onChange={(e) => setBaths(e.target.value)} required disabled={txPending} />
                        <input type="number" placeholder="Sqft" value={sqft} onChange={(e) => setSqft(e.target.value)} required disabled={txPending} />
                        <input type="number" placeholder="Year built" value={year} onChange={(e) => setYear(e.target.value)} required disabled={txPending} />
                    </div>

                    <div className="list__row">
                        <input type="number" step="0.1" placeholder="Price (ETH)" value={price} onChange={(e) => setPrice(e.target.value)} required disabled={txPending} />
                        <input type="number" step="0.1" placeholder="Escrow amount (ETH)" value={escrowAmount} onChange={(e) => setEscrowAmount(e.target.value)} required disabled={txPending} />
                    </div>

                    <div className={`list__field${sellerVerified ? ' list__field--verified' : ''}`}>
                        <input type="text" placeholder="Seller wallet address" value={sellerAddress} onChange={(e) => setSellerAddress(e.target.value)} required disabled={txPending} />
                        {sellerVerified && <span className="list__tick">&#10003;</span>}
                    </div>
                    {!sellerVerified && <p className="list__hint">Enter the wallet address connected to this app.</p>}

                    <label className="list__upload">
                        <span>{image ? 'Image selected' : 'Upload a photo'}</span>
                        <input type="file" accept="image/*" onChange={onImageChange} required disabled={txPending} />
                    </label>

                    <button type="submit" className="list__submit" disabled={txPending || !sellerVerified}>
                        {txPending ? 'Listing...' : sellerVerified ? 'List property' : 'Verify seller wallet'}
                    </button>
                </form>

                <button onClick={toggleList} className="home__close" disabled={txPending}>
                    <img src={close} alt="Close" />
                </button>
            </div>
        </div>
    );
}

export default List;
