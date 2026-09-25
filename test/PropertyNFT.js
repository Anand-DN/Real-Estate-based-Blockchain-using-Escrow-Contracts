const { expect } = require('chai');
const { ethers } = require('hardhat');

describe('PropertyNFT', () => {
    let admin, minter, other, user, operator
    let propertyNFT

    const PID = 'MREID_0000001'
    const URI = 'http://localhost:3000/metadata/millow/MREID_0000001.json'

    const missingRole = async (role, account) =>
        `AccessControl: account ${account.address.toLowerCase()} is missing role ${role.toLowerCase()}`

    beforeEach(async () => {
        [admin, minter, other, user, operator] = await ethers.getSigners()

        const PropertyNFT = await ethers.getContractFactory('PropertyNFT')
        propertyNFT = await PropertyNFT.deploy()

        await propertyNFT.grantRole(
            await propertyNFT.MINTER_ROLE(),
            minter.address,
        )
    })

    describe('Chain and deployment', () => {
        it('runs on the local development chain (chainId 31337)', async () => {
            const network = await ethers.provider.getNetwork()
            expect(network.chainId).to.equal(31337)
        })

        it('has the correct name and symbol', async () => {
            expect(await propertyNFT.name()).to.equal('MILLOW Property')
            expect(await propertyNFT.symbol()).to.equal('MILLOW')
        })

        it('supports ERC-165, ERC-721 and ERC-721 metadata interfaces', async () => {
            expect(await propertyNFT.supportsInterface('0x01ffc9a7')).to.equal(true)
            expect(await propertyNFT.supportsInterface('0x80ac58cd')).to.equal(true)
            expect(await propertyNFT.supportsInterface('0x5b5e139f')).to.equal(true)
            expect(await propertyNFT.supportsInterface('0xffffffff')).to.equal(false)
        })

        it('grants DEFAULT_ADMIN_ROLE and MINTER_ROLE to the deployer', async () => {
            const adminRole = await propertyNFT.DEFAULT_ADMIN_ROLE()
            const minterRole = await propertyNFT.MINTER_ROLE()
            expect(await propertyNFT.hasRole(adminRole, admin.address)).to.equal(true)
            expect(await propertyNFT.hasRole(minterRole, admin.address)).to.equal(true)
        })
    })

    describe('Minting and authorization', () => {
        it('allows a MINTER to mint a property', async () => {
            await expect(
                propertyNFT.connect(minter).mintProperty(PID, URI),
            ).to.emit(propertyNFT, 'PropertyMinted').withArgs(1, PID, minter.address)

            expect(await propertyNFT.ownerOf(1)).to.equal(minter.address)
            expect(await propertyNFT.totalSupply()).to.equal(1)
        })

        it('rejects minting from a non-minter account', async () => {
            await expect(
                propertyNFT.connect(other).mintProperty(PID, URI),
            ).to.be.revertedWith(
                await missingRole(await propertyNFT.MINTER_ROLE(), other),
            )
        })

        it('rejects an empty property id', async () => {
            await expect(
                propertyNFT.connect(minter).mintProperty('', URI),
            ).to.be.revertedWith('Property id is required')
        })

        it('prevents minting the same property twice', async () => {
            await propertyNFT.connect(minter).mintProperty(PID, URI)
            await expect(
                propertyNFT.connect(minter).mintProperty(PID, URI),
            ).to.be.revertedWith('Property already registered')
        })

        it('increments token ids per mint', async () => {
            await propertyNFT.connect(minter).mintProperty('MREID_0000001', URI)
            await propertyNFT.connect(minter).mintProperty('MREID_0000002', URI)
            expect(await propertyNFT.totalSupply()).to.equal(2)
            expect(await propertyNFT.tokenByProperty('MREID_0000002')).to.equal(2)
        })
    })

    describe('Property identity binding', () => {
        beforeEach(async () => {
            await propertyNFT.connect(minter).mintProperty(PID, URI)
        })

        it('maps property id to token id and back', async () => {
            expect(await propertyNFT.tokenByProperty(PID)).to.equal(1)
            expect(await propertyNFT.propertyOf(1)).to.equal(PID)
        })

        it('returns zero for an unknown property id', async () => {
            expect(await propertyNFT.tokenByProperty('MREID_9999999')).to.equal(0)
        })

        it('reverts propertyOf for a nonexistent token', async () => {
            await expect(propertyNFT.propertyOf(999)).to.be.revertedWith(
                'Token does not exist',
            )
        })
    })

    describe('Token URI and metadata', () => {
        beforeEach(async () => {
            await propertyNFT.connect(minter).mintProperty(PID, URI)
        })

        it('stores the provided token URI', async () => {
            expect(await propertyNFT.tokenURI(1)).to.equal(URI)
        })

        it('allows only MINTER to update a token URI', async () => {
            const next = 'http://localhost:3000/metadata/millow/MREID_0000001-v2.json'
            await expect(
                propertyNFT.connect(user).setTokenURI(1, next),
            ).to.be.revertedWith(
                await missingRole(await propertyNFT.MINTER_ROLE(), user),
            )

            await propertyNFT.connect(minter).setTokenURI(1, next)
            expect(await propertyNFT.tokenURI(1)).to.equal(next)
        })

        it('rejects URI updates for nonexistent tokens', async () => {
            await expect(
                propertyNFT.connect(minter).setTokenURI(999, URI),
            ).to.be.revertedWith('Token does not exist')
        })
    })

    describe('Transfers and ownership', () => {
        beforeEach(async () => {
            await propertyNFT.connect(minter).mintProperty(PID, URI)
        })

        it('rejects transfers from a non-owner', async () => {
            await expect(
                propertyNFT.connect(user).transferFrom(minter.address, user.address, 1),
            ).to.be.revertedWith('ERC721: caller is not token owner nor approved')
        })

        it('lets the owner transfer the token', async () => {
            await expect(
                propertyNFT.connect(minter).transferFrom(minter.address, other.address, 1),
            ).to.emit(propertyNFT, 'Transfer').withArgs(minter.address, other.address, 1)

            expect(await propertyNFT.ownerOf(1)).to.equal(other.address)
        })

        it('lets an approved operator transfer on behalf of the owner', async () => {
            await propertyNFT.connect(minter).setApprovalForAll(operator.address, true)
            await propertyNFT.connect(operator).transferFrom(minter.address, other.address, 1)
            expect(await propertyNFT.ownerOf(1)).to.equal(other.address)
        })
    })

    describe('Unrestricted minting resistance', () => {
        it('exposes no public open mint entry point', async () => {
            const functionNames = Object.keys(propertyNFT.interface.functions)
            expect(functionNames).to.not.include('mint(string)')
            expect(functionNames).to.not.include('mint()')
        })
    })
})