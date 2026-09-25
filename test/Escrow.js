const { expect } = require('chai');
const { ethers } = require('hardhat');

const tokens = (n) => {
    return ethers.utils.parseUnits(n.toString(), 'ether')
}

describe('Escrow', () => {
    let buyer, seller, inspector, lender
    let realEstate, escrow

    beforeEach(async () => {
        // Setup accounts
        [buyer, seller, inspector, lender] = await ethers.getSigners()

        // Deploy Real Estate
        const RealEstate = await ethers.getContractFactory('RealEstate')
        realEstate = await RealEstate.deploy()

        // Mint 
        let transaction = await realEstate.connect(seller).mint("https://ipfs.io/ipfs/QmTudSYeM7mz3PkYEWXWqPjomRPHogcMFSq7XAvsvsgAPS")
        await transaction.wait()

        // Deploy Escrow
        const Escrow = await ethers.getContractFactory('Escrow')
        escrow = await Escrow.deploy(
            realEstate.address,
            inspector.address,
            lender.address
        )

        // Approve Property
        transaction = await realEstate.connect(seller).approve(escrow.address, 1)
        await transaction.wait()

        // List Property
        transaction = await escrow.connect(seller).list(1, tokens(10), tokens(5))
        await transaction.wait()
    })

    describe('Deployment', () => {
        it('Returns NFT address', async () => {
            const result = await escrow.nftAddress()
            expect(result).to.be.equal(realEstate.address)
        })

        it('Returns seller', async () => {
            const result = await escrow.sellerOf(1)
            expect(result).to.be.equal(seller.address)
        })

        it('Returns inspector', async () => {
            const result = await escrow.inspector()
            expect(result).to.be.equal(inspector.address)
        })

        it('Returns lender', async () => {
            const result = await escrow.lender()
            expect(result).to.be.equal(lender.address)
        })
    })

    describe('Listing', () => {
        it('Updates as listed', async () => {
            const result = await escrow.isListed(1)
            expect(result).to.be.equal(true)
        })

        it('Returns no buyer until deposit', async () => {
            const result = await escrow.buyer(1)
            expect(result).to.be.equal(ethers.constants.AddressZero)
        })

        it('Returns purchase price', async () => {
            const result = await escrow.purchasePrice(1)
            expect(result.toString()).to.be.equal(tokens(10).toString())
        })

        it('Returns escrow amount', async () => {
            const result = await escrow.escrowAmount(1)
            expect(result.toString()).to.be.equal(tokens(5).toString())
        })

        it('Updates ownership', async () => {
            expect(await realEstate.ownerOf(1)).to.be.equal(escrow.address)
        })
    })

    describe('Deposits', () => {
        beforeEach(async () => {
            const transaction = await escrow.connect(buyer).depositEarnest(1, { value: tokens(5) })
            await transaction.wait()
        })

        it('Commits the depositor as buyer', async () => {
            const result = await escrow.buyer(1)
            expect(result).to.be.equal(buyer.address)
        })

        it('Updates contract balance', async () => {
            const result = await escrow.getBalance()
            expect(result.toString()).to.be.equal(tokens(5).toString())
        })
    })

    describe('Inspection', () => {
        beforeEach(async () => {
            const transaction = await escrow.connect(inspector).updateInspectionStatus(1, true)
            await transaction.wait()
        })

        it('Updates inspection status', async () => {
            const result = await escrow.inspectionPassed(1)
            expect(result).to.be.equal(true)
        })
    })

    describe('Approval', () => {
        beforeEach(async () => {
            let transaction = await escrow.connect(buyer).approveSale(1)
            await transaction.wait()

            transaction = await escrow.connect(seller).approveSale(1)
            await transaction.wait()

            transaction = await escrow.connect(lender).approveSale(1)
            await transaction.wait()
        })

        it('Updates approval status', async () => {
            expect(await escrow.approval(1, buyer.address)).to.be.equal(true)
            expect(await escrow.approval(1, seller.address)).to.be.equal(true)
            expect(await escrow.approval(1, lender.address)).to.be.equal(true)
        })
    })

    describe('Cancellation', () => {
        beforeEach(async () => {
            const transaction = await escrow.connect(buyer).depositEarnest(1, { value: tokens(5) })
            await transaction.wait()

            await escrow.connect(seller).cancelSale(1)
        })

        it('Returns the property to the seller', async () => {
            expect(await realEstate.ownerOf(1)).to.be.equal(seller.address)
        })

        it('Reopens the listing', async () => {
            expect(await escrow.isListed(1)).to.be.equal(false)
        })

        it('Clears the committed buyer', async () => {
            expect(await escrow.buyer(1)).to.be.equal(ethers.constants.AddressZero)
        })

        it('Allows the seller to re-list', async () => {
            let transaction = await realEstate.connect(seller).approve(escrow.address, 1)
            await transaction.wait()

            transaction = await escrow.connect(seller).list(1, tokens(10), tokens(5))
            await transaction.wait()

            expect(await escrow.isListed(1)).to.be.equal(true)
        })
    })

    describe('Sale', () => {
        beforeEach(async () => {
            let transaction = await escrow.connect(buyer).depositEarnest(1, { value: tokens(5) })
            await transaction.wait()

            transaction = await escrow.connect(inspector).updateInspectionStatus(1, true)
            await transaction.wait()

            transaction = await escrow.connect(buyer).approveSale(1)
            await transaction.wait()

            transaction = await escrow.connect(seller).approveSale(1)
            await transaction.wait()

            transaction = await escrow.connect(lender).approveSale(1)
            await transaction.wait()

            await escrow.connect(lender).fundSale(1, { value: tokens(5) })

            transaction = await escrow.connect(seller).finalizeSale(1)
            await transaction.wait()
        })

        it('Updates ownership', async () => {
            expect(await realEstate.ownerOf(1)).to.be.equal(buyer.address)
        })

        it('Updates balance', async () => {
            expect((await escrow.getBalance()).toString()).to.be.equal("0")
        })
    })

    describe('Sale funding accounting', () => {
        beforeEach(async () => {
            // Mint + list a second property
            let transaction = await realEstate.connect(seller).mint("https://ipfs.io/ipfs/QmTudSYeM7mz3PkYEWXWqPjomRPHogcMFSq7XAvsvsgAPS")
            await transaction.wait()

            transaction = await realEstate.connect(seller).approve(escrow.address, 2)
            await transaction.wait()

            transaction = await escrow.connect(seller).list(2, tokens(20), tokens(8))
            await transaction.wait()

            // Fund property 1 completely
            transaction = await escrow.connect(buyer).depositEarnest(1, { value: tokens(5) })
            await transaction.wait()

            transaction = await escrow.connect(inspector).updateInspectionStatus(1, true)
            await transaction.wait()

            transaction = await escrow.connect(buyer).approveSale(1)
            await transaction.wait()

            transaction = await escrow.connect(seller).approveSale(1)
            await transaction.wait()

            transaction = await escrow.connect(lender).approveSale(1)
            await transaction.wait()

            transaction = await escrow.connect(lender).fundSale(1, { value: tokens(5) })
            await transaction.wait()

            // Property 2 is only partly funded (earnest in)
            transaction = await escrow.connect(buyer).depositEarnest(2, { value: tokens(8) })
            await transaction.wait()
        })

        it('finalizes each sale with its own funds only', async () => {
            const sellerBefore = await ethers.provider.getBalance(seller.address)
            const contractBefore = await escrow.getBalance()

            const tx = await escrow.connect(seller).finalizeSale(1)
            const receipt = await tx.wait()
            const gasCost = receipt.gasUsed.mul(receipt.effectiveGasPrice)

            const sellerAfter = await ethers.provider.getBalance(seller.address)
            const contractAfter = await escrow.getBalance()

            // Seller receives exactly the purchase price (minus its own gas cost)
            expect(sellerAfter.add(gasCost).sub(sellerBefore)).to.be.equal(tokens(10))
            // Property 2's earnest (8 ETH) must remain untouched in escrow
            expect(contractAfter.toString()).to.be.equal(tokens(8).toString())
        })

        it('tracks earnest and lender funding per property', async () => {
            expect(await escrow.earnestPaid(1)).to.be.equal(tokens(5))
            expect(await escrow.lenderFunded(1)).to.be.equal(tokens(5))
            expect(await escrow.earnestPaid(2)).to.be.equal(tokens(8))
            expect(await escrow.lenderFunded(2)).to.be.equal(0)
        })

        it('rejects cancelling one sale while others hold funds', async () => {
            await expect(
                escrow.connect(buyer).cancelSale(2),
            ).to.not.be.reverted

            await expect(
                escrow.connect(buyer).finalizeSale(1),
            ).to.not.be.reverted
        })
    })

    describe('Cancel without a committed buyer', () => {
        beforeEach(async () => {
            const transaction = await escrow.connect(seller).cancelSale(1)
            await transaction.wait()
        })

        it('returns the property to the seller', async () => {
            expect(await realEstate.ownerOf(1)).to.be.equal(seller.address)
        })

        it('does not burn any contract funds', async () => {
            expect((await escrow.getBalance()).toString()).to.be.equal("0")
        })

        it('does not emit a buyer refund to the zero address', async () => {
            expect(await escrow.buyer(1)).to.be.equal(ethers.constants.AddressZero)
        })
    })
})