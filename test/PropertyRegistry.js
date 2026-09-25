const { expect } = require('chai');
const { ethers } = require('hardhat');

describe('PropertyRegistry', () => {
    let admin, minter, seller, buyer, stranger
    let propertyNFT, propertyRegistry

    const PID_A = 'MREID_0000001'
    const PID_B = 'MREID_0000002'
    const URI_A = 'http://localhost:3000/metadata/millow/MREID_0000001.json'
    const URI_B = 'http://localhost:3000/metadata/millow/MREID_0000002.json'

    const mintAndOwn = async (owner, pid, uri) => {
        await propertyNFT.connect(minter).mintProperty(pid, uri)
        const tokenId = await propertyNFT.tokenByProperty(pid)
        if (minter.address.toLowerCase() !== owner.address.toLowerCase()) {
            await propertyNFT
                .connect(minter)
                .transferFrom(minter.address, owner.address, tokenId)
        }
        return tokenId
    }

    const missingRole = async (role, account) =>
        `AccessControl: account ${account.address.toLowerCase()} is missing role ${role.toLowerCase()}`

    beforeEach(async () => {
        [admin, minter, seller, buyer, stranger] = await ethers.getSigners()

        const PropertyNFT = await ethers.getContractFactory('PropertyNFT')
        propertyNFT = await PropertyNFT.deploy()
        await propertyNFT.grantRole(await propertyNFT.MINTER_ROLE(), minter.address)

        const PropertyRegistry = await ethers.getContractFactory('PropertyRegistry')
        propertyRegistry = await PropertyRegistry.deploy(propertyNFT.address)
    })

    describe('Chain and deployment', () => {
        it('runs on the local development chain (chainId 31337)', async () => {
            const network = await ethers.provider.getNetwork()
            expect(network.chainId).to.equal(31337)
        })

        it('points at the PropertyNFT contract', async () => {
            expect(await propertyRegistry.propertyNFT()).to.equal(propertyNFT.address)
        })

        it('grants DEFAULT_ADMIN_ROLE and LISTING_MANAGER_ROLE to the deployer', async () => {
            const adminRole = await propertyRegistry.DEFAULT_ADMIN_ROLE()
            const managerRole = await propertyRegistry.LISTING_MANAGER_ROLE()
            expect(await propertyRegistry.hasRole(adminRole, admin.address)).to.equal(true)
            expect(await propertyRegistry.hasRole(managerRole, admin.address)).to.equal(true)
        })

        it('rejects a zero NFT address', async () => {
            const PropertyRegistry = await ethers.getContractFactory('PropertyRegistry')
            await expect(
                PropertyRegistry.deploy(ethers.constants.AddressZero),
            ).to.be.revertedWith('NFT address required')
        })
    })

    describe('Listing', () => {
        let tokenId

        beforeEach(async () => {
            tokenId = await mintAndOwn(seller, PID_A, URI_A)
        })

        it('lets the NFT owner list the token', async () => {
            await expect(
                propertyRegistry.connect(seller).list(tokenId),
            ).to.emit(propertyRegistry, 'Listed')

            expect(await propertyRegistry.isOnOffer(tokenId)).to.equal(true)
            expect(await propertyRegistry.sellerOf(tokenId)).to.equal(seller.address)
            expect(await propertyRegistry.activeListings()).to.equal(1)
        })

        it('rejects listing from a non-owner', async () => {
            await expect(
                propertyRegistry.connect(stranger).list(tokenId),
            ).to.be.revertedWith('Only the NFT owner can list')
        })

        it('rejects listing a token that does not exist', async () => {
            await expect(
                propertyRegistry.connect(seller).list(999),
            ).to.be.reverted
        })

        it('rejects listing the same token twice', async () => {
            await propertyRegistry.connect(seller).list(tokenId)
            await expect(
                propertyRegistry.connect(seller).list(tokenId),
            ).to.be.revertedWith('Already on offer')
        })
    })

    describe('Unlisting', () => {
        let tokenId

        beforeEach(async () => {
            tokenId = await mintAndOwn(seller, PID_A, URI_A)
            await propertyRegistry.connect(seller).list(tokenId)
        })

        it('lets the seller unlist their own token', async () => {
            await expect(
                propertyRegistry.connect(seller).unlist(tokenId),
            ).to.emit(propertyRegistry, 'Unlisted').withArgs(tokenId, seller.address)

            expect(await propertyRegistry.isOnOffer(tokenId)).to.equal(false)
            expect(await propertyRegistry.activeListings()).to.equal(0)
        })

        it('lets a LISTING_MANAGER unlist on behalf of a seller', async () => {
            await propertyRegistry.grantRole(await propertyRegistry.LISTING_MANAGER_ROLE(), stranger.address)

            await propertyRegistry.connect(stranger).unlist(tokenId)
            expect(await propertyRegistry.isOnOffer(tokenId)).to.equal(false)
        })

        it('rejects unlisting from a random account', async () => {
            await expect(
                propertyRegistry.connect(stranger).unlist(tokenId),
            ).to.be.revertedWith('Only the seller or a listing manager')
        })

        it('rejects unlisting a token that is not on offer', async () => {
            const otherToken = await mintAndOwn(seller, PID_B, URI_B)
            await expect(
                propertyRegistry.connect(seller).unlist(otherToken),
            ).to.be.revertedWith('Not on offer')
        })
    })

    describe('Multiple listings', () => {
        it('tracks activeListings count across tokens', async () => {
            const a = await mintAndOwn(seller, PID_A, URI_A)
            const b = await mintAndOwn(seller, PID_B, URI_B)

            await propertyRegistry.connect(seller).list(a)
            expect(await propertyRegistry.activeListings()).to.equal(1)

            await propertyRegistry.connect(seller).list(b)
            expect(await propertyRegistry.activeListings()).to.equal(2)

            await propertyRegistry.connect(seller).unlist(a)
            expect(await propertyRegistry.activeListings()).to.equal(1)
        })
    })

    describe('Pause', () => {
        let tokenId

        beforeEach(async () => {
            tokenId = await mintAndOwn(seller, PID_A, URI_A)
        })

        it('lets only the admin pause the registry', async () => {
            await expect(
                propertyRegistry.connect(stranger).pause(),
            ).to.be.revertedWith(
                await missingRole(await propertyRegistry.DEFAULT_ADMIN_ROLE(), stranger),
            )

            await propertyRegistry.connect(admin).pause()
        })

        it('blocks listing while paused', async () => {
            await propertyRegistry.connect(admin).pause()
            await expect(
                propertyRegistry.connect(seller).list(tokenId),
            ).to.be.reverted
        })

        it('blocks unlisting while paused', async () => {
            await propertyRegistry.connect(seller).list(tokenId)
            await propertyRegistry.connect(admin).pause()
            await expect(
                propertyRegistry.connect(seller).unlist(tokenId),
            ).to.be.reverted
        })

        it('lets only the admin unpause', async () => {
            await propertyRegistry.connect(admin).pause()
            await expect(
                propertyRegistry.connect(stranger).unpause(),
            ).to.be.revertedWith(
                await missingRole(await propertyRegistry.DEFAULT_ADMIN_ROLE(), stranger),
            )

            await propertyRegistry.connect(admin).unpause()
            await propertyRegistry.connect(seller).list(tokenId)
            expect(await propertyRegistry.isOnOffer(tokenId)).to.equal(true)
        })
    })

    describe('Ownership source of truth', () => {
        it('derives listing eligibility from PropertyNFT.ownerOf', async () => {
            const tokenId = await mintAndOwn(seller, PID_A, URI_A)

            await propertyRegistry.connect(seller).list(tokenId)
            expect(await propertyRegistry.sellerOf(tokenId)).to.equal(seller.address)

            await propertyNFT.connect(seller).transferFrom(seller.address, buyer.address, tokenId)

            await propertyRegistry.connect(seller).unlist(tokenId)
            await propertyRegistry.connect(buyer).list(tokenId)
            expect(await propertyRegistry.sellerOf(tokenId)).to.equal(buyer.address)
        })
    })
})