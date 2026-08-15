import json
import os
import re
import glob

import numpy as np
import joblib

from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

BASE_DIR = os.path.dirname(__file__)
META_DIR = os.path.join(BASE_DIR, '..', 'public', 'metadata')
MODEL_DIR = os.path.join(BASE_DIR, 'models')
PROPS_CACHE = os.path.join(BASE_DIR, 'properties.json')
RANDOM_STATE = 42

PROPERTY_TYPES = ['Condo', 'Single family residence', 'Townhouse', 'Cabin']


def extract_region(address):
    match = re.search(r',\s*([A-Z]{2})\s*\d{5}', address or '')
    return match.group(1) if match else 'Other'


def load_properties():
    properties = []
    for path in sorted(
        glob.glob(os.path.join(META_DIR, '*.json')),
        key=lambda p: int(os.path.splitext(os.path.basename(p))[0]),
    ):
        with open(path, encoding='utf-8') as handle:
            metadata = json.load(handle)
        attrs = {a['trait_type']: a['value'] for a in metadata.get('attributes', [])}
        properties.append({
            'token_id': int(metadata.get('id')),
            'name': metadata.get('name'),
            'address': metadata.get('address'),
            'description': metadata.get('description', ''),
            'image': metadata.get('image'),
            'region': extract_region(metadata.get('address')),
            'property_type': attrs.get('Type of Residence', 'Single family residence'),
            'beds': int(attrs.get('Bed Rooms', 0)),
            'baths': int(attrs.get('Bathrooms', 0)),
            'sqft': float(attrs.get('Square Feet', 0)),
            'year_built': int(attrs.get('Year Built', 0)),
            'price_eth': float(attrs.get('Purchase Price', 0)),
        })
    return properties


def price_feature_columns(regions):
    return (['beds', 'baths', 'sqft', 'year_built']
            + [f'type_{t}' for t in PROPERTY_TYPES]
            + [f'region_{r}' for r in regions])


def featurize_price(props, regions):
    rows = []
    for p in props:
        row = [p['beds'], p['baths'], p['sqft'], p['year_built']]
        row += [1 if p['property_type'] == t else 0 for t in PROPERTY_TYPES]
        row += [1 if p['region'] == r else 0 for r in regions]
        rows.append(row)
    return np.asarray(rows, dtype=float)


def build_price_model(properties, regions):
    cols = price_feature_columns(regions)
    real_props = [p for p in properties if p['sqft'] > 0 and p['price_eth'] > 0]
    n_real = len(real_props)

    base_cols = ['beds', 'baths', 'log_sqft', 'year_norm']
    base_cols += [f'type_{t}' for t in PROPERTY_TYPES]
    base_cols += [f'region_{r}' for r in regions]

    def base_vector(p):
        return [
            p['beds'],
            p['baths'],
            np.log(p['sqft']),
            p['year_built'] - 1985,
        ] + [1 if p['property_type'] == t else 0 for t in PROPERTY_TYPES] \
            + [1 if p['region'] == r else 0 for r in regions]

    X_real = np.asarray([base_vector(p) for p in real_props], dtype=float)
    y_real = np.log(np.asarray([p['price_eth'] for p in real_props], dtype=float))
    X_real = np.column_stack([np.ones(n_real), X_real])

    identity = np.eye(X_real.shape[1])
    identity[0, 0] = 0
    beta = np.linalg.solve(X_real.T @ X_real + 2.0 * identity, X_real.T @ y_real)
    residual_std = float(np.std(y_real - X_real @ beta))

    rng = np.random.default_rng(RANDOM_STATE)
    weights_type = np.asarray([sum(1 for p in real_props if p['property_type'] == t) for t in PROPERTY_TYPES], dtype=float)
    weights_type /= weights_type.sum()
    weights_region = np.asarray([sum(1 for p in real_props if p['region'] == r) for r in regions], dtype=float)
    weights_region /= weights_region.sum()

    synthetic = []
    for _ in range(4000):
        beds = int(rng.choice([1, 2, 3, 4, 5, 6], p=[0.05, 0.20, 0.35, 0.25, 0.10, 0.05]))
        baths = max(1, int(rng.normal(beds * 0.85, 0.7)))
        sqft = float(np.clip(400 + beds * 500 + rng.normal(0, 280), 700, 8000))
        year = int(rng.integers(1890, 2026))
        ptype = str(rng.choice(PROPERTY_TYPES, p=weights_type))
        region = str(rng.choice(regions, p=weights_region))
        proto = {'beds': beds, 'baths': baths, 'sqft': sqft, 'year_built': year,
                 'property_type': ptype, 'region': region}
        log_price = float(np.asarray([1] + base_vector(proto)) @ beta) + rng.normal(0, residual_std)
        synthetic.append({
            'beds': beds, 'baths': baths, 'sqft': sqft, 'year_built': year,
            'property_type': ptype, 'region': region,
            'price_eth': float(np.exp(log_price)),
        })

    X = np.asarray([[s[c] for c in ['beds', 'baths', 'sqft', 'year_built']]
                    + [1 if s['property_type'] == t else 0 for t in PROPERTY_TYPES]
                    + [1 if s['region'] == r else 0 for r in regions] for s in synthetic], dtype=float)
    y = np.asarray([s['price_eth'] for s in synthetic], dtype=float)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)
    model = RandomForestRegressor(n_estimators=250, max_depth=14, random_state=RANDOM_STATE, n_jobs=-1)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mae = float(np.mean(np.abs(y_pred - y_test)))
    rmse = float(np.sqrt(np.mean((y_pred - y_test) ** 2)))
    r2 = float(model.score(X_test, y_test))
    print(f'[price] real listings used: {n_real} | synthetic samples: {len(synthetic)}')
    print(f'[price] holdout MAE={mae:.3f} ETH RMSE={rmse:.3f} ETH R2={r2:.3f}')

    return {'model': model, 'feature_cols': cols, 'regions': regions}


def build_fraud_model():
    rng = np.random.default_rng(RANDOM_STATE)
    fraud_cols = [
        'escrow_ratio', 'time_between_hours', 'cancel_count', 'failed_count',
        'listed_vs_pred_ratio', 'history_count', 'rapid_activity',
    ]

    def sample_normal():
        return [
            float(np.clip(rng.normal(0.10, 0.03), 0.03, 0.50)),
            float(rng.exponential(96.0)),
            float(min(int(rng.poisson(0.15)), 3)),
            float(min(int(rng.poisson(0.3)), 4)),
            float(rng.normal(1.0, 0.12)),
            float(min(int(rng.gamma(3.0, 3.0)), 60)),
            0.0,
        ]

    def sample_anomaly():
        return [
            float(rng.uniform(0.55, 0.90)),
            float(rng.uniform(0.5, 8.0)),
            float(rng.integers(2, 4)),
            float(rng.integers(2, 5)),
            float(rng.uniform(0.45, 1.65)),
            float(rng.integers(0, 5)),
            1.0,
        ]

    rows = [sample_normal() for _ in range(4000)]
    rows += [sample_anomaly() for _ in range(210)]
    X = np.asarray(rows, dtype=float)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = IsolationForest(n_estimators=200, contamination=0.05, random_state=RANDOM_STATE, n_jobs=-1)
    model.fit(X_scaled)

    raw_scores = model.decision_function(X_scaled)
    score_min = float(np.percentile(raw_scores, 1))
    score_max = float(np.percentile(raw_scores, 99))

    print(f'[fraud] synthetic transactions: {len(rows)} (5% injected anomalies)')
    return {'model': model, 'scaler': scaler, 'feature_cols': fraud_cols,
            'score_min': score_min, 'score_max': score_max}


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    properties = load_properties()
    regions = sorted({p['region'] for p in properties})
    print(f'[data] loaded {len(properties)} properties, regions={regions}')

    with open(PROPS_CACHE, 'w', encoding='utf-8') as handle:
        json.dump(properties, handle, indent=2)

    price_artifact = build_price_model(properties, regions)
    joblib.dump(price_artifact, os.path.join(MODEL_DIR, 'price_model.joblib'))

    fraud_artifact = build_fraud_model()
    joblib.dump(fraud_artifact, os.path.join(MODEL_DIR, 'fraud_model.joblib'))

    print('[done] models saved to ai/models/')


if __name__ == '__main__':
    main()
