import PropertyThumb from "./PropertyThumb";
import { formatInr } from "../lib/format";

const PropertyCard = ({ property, onSelect }) => {
  const amenityCounts = property.amenities || {};
  const yesCount = Object.values(amenityCounts).filter((v) => v === "Yes").length;

  return (
    <article
      className="mkt__card"
      onClick={() => onSelect(property.mreid_id)}
      tabIndex={0}
      role="button"
      aria-label={`${property.location}, ${property.city} — listed ${property.price_formatted}`}
      onKeyDown={(e) => {
        if (e.key === "Enter" || e.key === " ") {
          e.preventDefault();
          onSelect(property.mreid_id);
        }
      }}
    >
      <PropertyThumb mreidId={property.mreid_id} city={property.city} compact />

      <div className="mkt__card-body">
        <span className="mkt__card-city">{property.city}</span>
        <h3 className="mkt__card-location">{property.location}</h3>
        <p className="mkt__card-meta">
          <strong>{property.bedrooms}</strong> bds ·{" "}
          <strong>{Number(property.area).toLocaleString("en-IN")}</strong> sqft
        </p>

        <div className="mkt__card-price">
          <span className="mkt__label">Listed price</span>
          <span className="mkt__listed">{property.price_formatted}</span>
        </div>

        <div className="mkt__card-ai">
          <span className="mkt__label">
            MILLOW AI estimate
            <span className="mkt__ai-badge">AI</span>
          </span>
          <span className="mkt__ai-value">
            {property.ai_estimated_price_formatted}
          </span>
          <span className="mkt__ai-sqft">
            ≈ {formatInr(property.ai_estimated_price_per_sqft)}/sqft
          </span>
        </div>

        <div className="mkt__card-foot">
          <span>{yesCount} amenities</span>
          <span className="mkt__card-view">View details →</span>
        </div>
      </div>
    </article>
  );
};

export default PropertyCard;