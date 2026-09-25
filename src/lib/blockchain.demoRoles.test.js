import { ethers } from "ethers";
import config from "../config.json";
import {
  ROLE_NAMES,
  demoRoleOf,
  resolveAccountRole,
} from "./blockchain";

// The canonical demo accounts must always resolve to their configured
// workspace role while a registered sale is open — never to Viewer.
const demo = config.demoRoles;
const INSPECTOR_ROLE = ethers.utils.id("INSPECTOR_ROLE");
const LENDER_ROLE = ethers.utils.id("LENDER_ROLE");

const noRolesEscrow = {
  hasRole: jest.fn(async () => false),
};

describe("demoRoleOf (config.json demoRoles)", () => {
  it("resolves the configured seller address", () => {
    expect(demoRoleOf(demo.seller)).toBe("seller");
  });

  it("resolves the configured buyer address", () => {
    expect(demoRoleOf(demo.buyer)).toBe("buyer");
  });

  it("resolves the configured inspector address", () => {
    expect(demoRoleOf(demo.inspector)).toBe("inspector");
  });

  it("resolves the configured lender address", () => {
    expect(demoRoleOf(demo.lender)).toBe("lender");
  });

  it("is case-insensitive on the address", () => {
    expect(demoRoleOf(String(demo.buyer).toUpperCase())).toBe("buyer");
  });

  it("returns null for unknown accounts and for no account", () => {
    expect(demoRoleOf("0x15d34AAf54267DB7D7c367839AAf71A00a2C6A65")).toBeNull();
    expect(demoRoleOf(null)).toBeNull();
    expect(demoRoleOf("")).toBeNull();
  });
});

describe("resolveAccountRole on-chain facts first", () => {
  // Matches the live demo chain: the escrow sale's parties ARE the configured
  // demo accounts.  foreignSale is a sale fed by accounts that are not demo
  // identities, so badges can never fake a role.
  const demoSale = {
    buyer: demo.buyer,
    seller: demo.seller,
  };
  const foreignSale = {
    buyer: "0x1111111111111111111111111111111111111111",
    seller: "0x2222222222222222222222222222222222222222",
  };

  it("wins the committed on-chain buyer over any demo badge", async () => {
    // The demo buyer is the escrow's committed buyer, so the buyer role comes
    // from the chain even though someone else owns the NFT.
    await expect(
      resolveAccountRole(noRolesEscrow, demoSale, demo.buyer, demo.seller),
    ).resolves.toBe("buyer");
  });

  it("wins the escrow sale seller over the demo badge and over ownership", async () => {
    await expect(
      resolveAccountRole(noRolesEscrow, demoSale, demo.seller),
    ).resolves.toBe("seller");
    // The sale self declaration wins even when someone else owns the NFT.
    await expect(
      resolveAccountRole(
        noRolesEscrow,
        demoSale,
        demo.seller,
        "0x7777777777777777777777777777777777777777",
      ),
    ).resolves.toBe("seller");
  });

  it("resolves the demo seller to Seller only when they actually own the NFT", async () => {
    await expect(
      resolveAccountRole(noRolesEscrow, foreignSale, demo.seller, demo.seller),
    ).resolves.toBe("seller");
  });

  it("never lets the demo seller badge override real on-chain ownership", async () => {
    // The demo seller is NOT the NFT owner and not a sale party here — the
    // badge must not fake a Seller role.
    await expect(
      resolveAccountRole(
        noRolesEscrow,
        foreignSale,
        demo.seller,
        "0x7777777777777777777777777777777777777777",
      ),
    ).resolves.toBe("viewer");
  });

  it("never lets the demo buyer badge override on-chain ownership", async () => {
    await expect(
      resolveAccountRole(
        noRolesEscrow,
        foreignSale,
        demo.buyer,
        "0x7777777777777777777777777777777777777777",
      ),
    ).resolves.toBe("viewer");
  });

  it("maps the actual NFT owner to Seller even without a configured demo role", async () => {
    const unknown = "0x15d34AAf54267DB7D7c367839AAf71A00a2C6A65";
    await expect(
      resolveAccountRole(noRolesEscrow, foreignSale, unknown, unknown),
    ).resolves.toBe("seller");
  });

  it("calls hasRole for Inspector/Lender assignment on the escrow", async () => {
    const spy = jest.fn(async () => false);
    await expect(
      resolveAccountRole({ hasRole: spy }, foreignSale, demo.inspector),
    ).resolves.toBe("viewer");
    expect(spy).toHaveBeenCalledWith(INSPECTOR_ROLE, demo.inspector);
    expect(spy).toHaveBeenCalledWith(LENDER_ROLE, demo.inspector);
  });

  it("maps the demo Inspector / Lender accounts through their escrow grants", async () => {
    const inspectorEscrow = {
      hasRole: async (role) => role === INSPECTOR_ROLE,
    };
    const lenderEscrow = {
      hasRole: async (role) => role === LENDER_ROLE,
    };
    await expect(
      resolveAccountRole(inspectorEscrow, foreignSale, demo.inspector),
    ).resolves.toBe("inspector");
    await expect(
      resolveAccountRole(lenderEscrow, foreignSale, demo.lender),
    ).resolves.toBe("lender");
    await expect(
      resolveAccountRole(noRolesEscrow, foreignSale, demo.inspector),
    ).resolves.toBe("viewer");
  });

  it("does not call the escrow when the NFT owner decides the seller role", async () => {
    await resolveAccountRole(noRolesEscrow, foreignSale, demo.seller, demo.seller);
    expect(noRolesEscrow.hasRole).not.toHaveBeenCalled();
  });

  it("falls back to Viewer for an unknown account with no party, ownership, or escrow role", async () => {
    const unknown = "0x15d34AAf54267DB7D7c367839AAf71A00a2C6A65";
    await expect(
      resolveAccountRole(noRolesEscrow, foreignSale, unknown),
    ).resolves.toBe("viewer");
  });

  it("derives the party roles for unknown accounts when they are on-chain buyer or seller", async () => {
    const unknown = "0x15d34AAf54267DB7D7c367839AAf71A00a2C6A65";
    await expect(
      resolveAccountRole(noRolesEscrow, { ...foreignSale, buyer: unknown }, unknown),
    ).resolves.toBe("buyer");
    await expect(
      resolveAccountRole(noRolesEscrow, { ...foreignSale, seller: unknown }, unknown),
    ).resolves.toBe("seller");
  });

  it("derives the escrow role for unknown accounts from hasRole", async () => {
    const unknown = "0x15d34AAf54267DB7D7c367839AAf71A00a2C6A65";
    const inspectorEscrow = {
      hasRole: async (role) => role === INSPECTOR_ROLE,
    };
    const lenderEscrow = {
      hasRole: async (role) => role === LENDER_ROLE,
    };
    await expect(
      resolveAccountRole(inspectorEscrow, foreignSale, unknown),
    ).resolves.toBe("inspector");
    await expect(
      resolveAccountRole(lenderEscrow, foreignSale, unknown),
    ).resolves.toBe("lender");
  });

  it("exposes every role through ROLE_NAMES for the badge", () => {
    expect(ROLE_NAMES).toMatchObject({
      seller: "Seller",
      buyer: "Buyer",
      inspector: "Inspector",
      lender: "Lender",
      viewer: "Viewer",
    });
  });

  it("no longer maps any Registrar role", () => {
    expect(ROLE_NAMES.registrar).toBeUndefined();
    expect(demoRoleOf(demo.seller)).not.toBe("registrar");
  });
});