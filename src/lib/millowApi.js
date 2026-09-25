const BASE =
  process.env.REACT_APP_PROPERTY_API_URL || "http://localhost:8001";

async function request(path) {
  let response;
  try {
    response = await fetch(`${BASE}${path}`, {
      headers: { Accept: "application/json" },
    });
  } catch (err) {
    throw new Error(
      `Could not reach the MILLOW property API at ${BASE}. Is the backend running (npm run valuation)?`,
    );
  }

  if (!response.ok) {
    let detail = `Request failed (${response.status})`;
    try {
      const data = await response.json();
      if (data && data.detail) {
        detail = typeof data.detail === "string" ? data.detail : JSON.stringify(data.detail);
      }
    } catch (_) {
      // keep generic message
    }
    throw new Error(detail);
  }

  return response.json();
}

export function cities() {
  return request("/api/properties/cities").then((d) => d.cities || []);
}

export function locations(city) {
  const qs = city ? `?city=${encodeURIComponent(city)}` : "";
  return request(`/api/properties/locations${qs}`).then((d) => d.locations || []);
}

export function propertyById(mreidId) {
  return request(`/api/properties/${encodeURIComponent(mreidId)}`);
}

export function propertyChain(mreidId) {
  return request(`/api/properties/${encodeURIComponent(mreidId)}/chain`);
}

export function marketContext(mreidId) {
  return request(`/api/properties/${encodeURIComponent(mreidId)}/market-context`);
}

export function riskAnalysis(mreidId) {
  return request(`/api/properties/${encodeURIComponent(mreidId)}/risk-analysis`);
}

export function recommendations(mreidId, options) {
  const qs = new URLSearchParams();
  if (options && options.limit) qs.append("limit", String(options.limit));
  const suffix = qs.toString() ? `?${qs.toString()}` : "";
  return request(
    `/api/properties/${encodeURIComponent(mreidId)}/recommendations${suffix}`,
  );
}

export function propertyRecommendations(params) {
  const qs = new URLSearchParams();
  Object.entries(params || {}).forEach(([key, value]) => {
    if (value !== undefined && value !== null && value !== "") {
      if (Array.isArray(value)) {
        if (value.length) qs.append(key, value.join(","));
      } else {
        qs.append(key, String(value));
      }
    }
  });
  return request(`/api/recommendations?${qs.toString()}`);
}

export function searchProperties(params) {
  const qs = new URLSearchParams();
  Object.entries(params || {}).forEach(([key, value]) => {
    if (value !== undefined && value !== null && value !== "") {
      qs.append(key, String(value));
    }
  });
  return request(`/api/properties/search?${qs.toString()}`);
}

export function dashboardOverview() {
  return request("/api/dashboard/overview");
}

export function marketBreakdown(params) {
  const qs = new URLSearchParams();
  Object.entries(params || {}).forEach(([key, value]) => {
    if (value !== undefined && value !== null && value !== "") {
      qs.append(key, String(value));
    }
  });
  const suffix = qs.toString() ? `?${qs.toString()}` : "";
  return request(`/api/dashboard/market-breakdown${suffix}`);
}

export function dashboardInsights() {
  return request("/api/dashboard/insights");
}