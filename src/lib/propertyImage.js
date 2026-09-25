const ILLUSTRATIVE_LABEL = "Illustrative image";
const IMAGE_COUNT = 24;

// FNV-1a: stable, deterministic hash for any MREID string.  Never random, so
// the same MREID always resolves to the same illustrative image.
const fnv1a = (input) => {
  const text = String(input || "");
  let hash = 0x811c9dc5;
  for (let i = 0; i < text.length; i++) {
    hash ^= text.charCodeAt(i);
    hash = Math.imul(hash, 0x01000193);
  }
  return hash >>> 0;
};

const mreidOf = (property) => {
  if (property == null) return "";
  if (typeof property === "string") return property;
  return (
    (property && (property.mreid_id || property.mreidId || property.id)) || ""
  );
};

export const imageLabel = ILLUSTRATIVE_LABEL;

// Deterministic 1-based image index for a property (1..IMAGE_COUNT).
export function propertyImageIndex(property) {
  const mreid = mreidOf(property);
  if (!mreid) return 1;
  return (fnv1a(mreid) % IMAGE_COUNT) + 1;
}

export function getPropertyImageUrl(property) {
  return `/images/${propertyImageIndex(property)}.jpg`;
}

export function getPropertyImage(property) {
  return {
    url: getPropertyImageUrl(property),
    label: ILLUSTRATIVE_LABEL,
    mreid: mreidOf(property),
  };
}

export default getPropertyImage;