import { ethers } from "ethers";
import config from "../config.json";

const INR_PER_ETH = Number((config.demoRate && config.demoRate.inrPerEth) || 100000);

export const demoRateInrPerEth = INR_PER_ETH;

export const demoRateLabel = `₹${INR_PER_ETH.toLocaleString("en-IN")} = 1 test ETH`;

export function inrToWei(inrRupees) {
  const whole = ethers.BigNumber.from(Math.round(Number(inrRupees)));
  return whole.mul(ethers.utils.parseEther("1")).div(ethers.BigNumber.from(INR_PER_ETH));
}