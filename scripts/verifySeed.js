const hre = require("hardhat");
const fs = require("fs");
const path = require("path");

async function main() {
  const config = JSON.parse(
    fs.readFileSync(path.join(__dirname, "..", "src", "config.json"), "utf8"),
  );
  const nftAddress = config["31337"].propertyNft.address;
  const seller = config.demoRoles.seller.toLowerCase();
  const NFT = await ethers.getContractFactory("PropertyNFT");
  const nft = NFT.attach(nftAddress);

  console.log("PropertyNFT:", nftAddress);
  console.log("Seller (config.demoRoles.seller):", seller);

  for (const id of ["MREID_0000001", "MREID_0000002"]) {
    const tok = (await nft.tokenByProperty(id)).toNumber();
    const owner = tok ? await nft.ownerOf(tok) : null;
    console.log(`tokenByProperty(${id}) =`, tok, tok ? "(!=0 OK)" : "(!! ZERO)");
    if (tok)
      console.log(
        `ownerOf(${tok}) =`,
        owner,
        owner && owner.toLowerCase() === seller ? "== Seller OK" : "(!! NOT SELLER)",
      );
  }
  console.log("totalSupply():", (await nft.totalSupply()).toString());
}

main().catch((e) => {
  console.error(e);
  process.exitCode = 1;
});