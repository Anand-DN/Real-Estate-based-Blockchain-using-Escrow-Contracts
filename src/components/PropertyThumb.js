import { getPropertyImageUrl, getPropertyImage, imageLabel } from "../lib/propertyImage";

const PropertyThumb = ({ mreidId, city, compact }) => {
  const src = getPropertyImageUrl(mreidId);
  const label = imageLabel;
  const icon = getPropertyImage(mreidId);

  return (
    <div
      className="mkt__thumb"
      data-compact={compact ? "true" : "false"}
      role="img"
      aria-label={`${label} for ${mreidId || "listing"}${city ? ` in ${city}` : ""}`}
    >
      <img
        className="mkt__thumb-img"
        src={src}
        alt={`${label} for ${mreidId || "listing"}`}
      />
      <span className="mkt__thumb-id">{icon.mreid || ""}</span>
      <span className="mkt__thumb-label">{label}</span>
    </div>
  );
};

export default PropertyThumb;