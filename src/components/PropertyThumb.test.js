import { render, screen } from "@testing-library/react";
import PropertyThumb from "./PropertyThumb";
import {
  getPropertyImage,
  getPropertyImageUrl,
  propertyImageIndex,
  imageLabel,
} from "../lib/propertyImage";

describe("propertyImage resolver", () => {
  it("maps each MREID to a stable, distinct illustrative image", () => {
    const one = getPropertyImageUrl("MREID_0000001");
    const two = getPropertyImageUrl("MREID_0000002");
    const three = getPropertyImageUrl("MREID_0000003");

    expect(one).toBe(getPropertyImageUrl("MREID_0000001"));
    expect(two).toBe(getPropertyImageUrl("MREID_0000002"));
    expect(three).toBe(getPropertyImageUrl("MREID_0000003"));
    expect(one).toMatch(/^\/images\/\d+\.jpg$/);
    expect(two).toMatch(/^\/images\/\d+\.jpg$/);
    expect(three).toMatch(/^\/images\/\d+\.jpg$/);
  });

  it("accepts property objects via mreid_id or mreidId", () => {
    const fromId = getPropertyImageUrl({ mreid_id: "MREID_0000224" });
    const fromAlt = getPropertyImageUrl({ mreidId: "MREID_0004390" });
    expect(fromId).toBe(getPropertyImageUrl("MREID_0000224"));
    expect(fromAlt).toBe(getPropertyImageUrl("MREID_0004390"));
  });

  it("never returns an index outside the image collection", () => {
    for (let i = 1; i <= 200; i++) {
      const idx = propertyImageIndex(`MREID_${String(i).padStart(7, "0")}`);
      expect(idx).toBeGreaterThanOrEqual(1);
      expect(idx).toBeLessThanOrEqual(24);
    }
  });

  it("exposes the illustrative label", () => {
    expect(imageLabel).toBe("Illustrative image");
    expect(getPropertyImage("MREID_0000001").label).toBe("Illustrative image");
  });
});

describe("PropertyThumb rendering", () => {
  it("renders an image with the illustrative label and no placeholder text", () => {
    render(<PropertyThumb mreidId="MREID_0000001" city="Bangalore" />);

    const img = screen.getByAltText(/Illustrative image for MREID_0000001/);
    expect(img).toBeInTheDocument();
    expect(img.tagName).toBe("IMG");
    expect(img).toHaveAttribute(
      "src",
      getPropertyImageUrl("MREID_0000001"),
    );

    expect(screen.getByText("Illustrative image")).toBeInTheDocument();
    expect(screen.getByText("MREID_0000001")).toBeInTheDocument();

    expect(
      screen.queryByText("Generic placeholder — no photo in dataset"),
    ).not.toBeInTheDocument();
    expect(screen.queryByText(/Placeholder/)).not.toBeInTheDocument();
  });

  it("renders distinct images for different MREIDs", () => {
    const { rerender } = render(
      <PropertyThumb mreidId="MREID_0000224" compact />,
    );
    const first = screen
      .getByAltText(/Illustrative image for MREID_0000224/)
      .getAttribute("src");

    rerender(<PropertyThumb mreidId="MREID_0004390" compact />);
    const second = screen
      .getByAltText(/Illustrative image for MREID_0004390/)
      .getAttribute("src");

    expect(first).toBe(getPropertyImageUrl("MREID_0000224"));
    expect(second).toBe(getPropertyImageUrl("MREID_0004390"));
  });
});