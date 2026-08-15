import json
import os

import numpy as np
import joblib
import uvicorn

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

import train

BASE_DIR = os.path.dirname(__file__)
MODEL_DIR = os.path.join(BASE_DIR, 'models')
PROPS_CACHE = os.path.join(BASE_DIR, 'properties.json')

app = FastAPI(title='Millow AI Intelligence', version='1.0.0')

app.add_middleware(
    CORSMiddleware,
    allow_origins=['http://localhost:3000', 'http://localhost:3001'],
    allow_origin_regex=r'http://localhost:\d+',
    allow_methods=['*'],
    allow_headers=['*'],
)


def load_or_train(name, builder):
    path = os.path.join(MODEL_DIR, name)
    if os.path.exists(path):
        return joblib.load(path)
    return builder()


def load_properties():
    if os.path.exists(PROPS_CACHE):
        with open(PROPS_CACHE, encoding='utf-8') as handle:
            return json.load(handle)
    props = train.load_properties()
    with open(PROPS_CACHE, 'w', encoding='utf-8') as handle:
        json.dump(props, handle, indent=2)
    return props


PRICE_MODEL = load_or_train('price_model.joblib', lambda: train.build_price_model(load_properties(), sorted({p['region'] for p in load_properties()})))
FRAUD_MODEL = load_or_train('fraud_model.joblib', train.build_fraud_model)
PROPERTIES = load_properties()

REC_COLS = ['beds', 'baths', 'log_sqft', 'year_built', 'price_eth',
            'type_Condo', 'type_Single family residence', 'type_Townhouse', 'type_Cabin']

REC_SCALER = StandardScaler()


def rec_vector(p):
    return [p['beds'], p['baths'], np.log(max(p['sqft'], 1)), p['year_built'], p['price_eth'],
            1 if p['property_type'] == 'Condo' else 0,
            1 if p['property_type'] == 'Single family residence' else 0,
            1 if p['property_type'] == 'Townhouse' else 0,
            1 if p['property_type'] == 'Cabin' else 0]


REC_MATRIX = REC_SCALER.fit_transform(np.asarray([rec_vector(p) for p in PROPERTIES], dtype=float))
REC_SIM = cosine_similarity(REC_MATRIX)


def price_feature_vector(p):
    cols = PRICE_MODEL['feature_cols']
    regions = PRICE_MODEL['regions']
    row = [p['beds'], p['baths'], p['sqft'], p['year_built']]
    row += [1 if p['property_type'] == t else 0 for t in train.PROPERTY_TYPES]
    row += [1 if p['region'] == r else 0 for r in regions]
    return np.asarray([row], dtype=float)


def predict_price_eth(p):
    x = price_feature_vector(p)
    pred = float(PRICE_MODEL['model'].predict(x)[0])
    return round(max(pred, 0.1), 2)


def fraud_vector(profile):
    cols = FRAUD_MODEL['feature_cols']
    return np.asarray([profile[c] for c in cols], dtype=float).reshape(1, -1)


def anomaly_score(profile):
    x = fraud_vector(profile)
    x_scaled = FRAUD_MODEL['scaler'].transform(x)
    raw = float(FRAUD_MODEL['model'].decision_function(x_scaled)[0])
    lo = FRAUD_MODEL['score_min']
    hi = FRAUD_MODEL['score_max']
    score = float(np.clip((hi - raw) / (hi - lo), 0.0, 1.0))
    return round(score, 3)


def verdict_for(diff_pct):
    if diff_pct > 5:
        return 'Undervalued'
    if diff_pct < -5:
        return 'Overvalued'
    return 'Fairly priced'


def compute_risk(predicted, listed, year_built, anomaly, buyer_risk=10.0, seller_risk=10.0):
    diff_ratio = abs(predicted - listed) / listed if listed else 0.5
    age_risk = float(np.clip((2026 - year_built) / 120, 0.0, 1.0))
    property_risk = float(np.clip(100 * (0.7 * diff_ratio / 0.5 + 0.3 * age_risk), 0.0, 100.0))
    anomaly_risk = float(np.clip(100 * anomaly, 0.0, 100.0))
    overall = round(0.30 * property_risk + 0.20 * buyer_risk + 0.20 * seller_risk + 0.30 * anomaly_risk, 1)
    level = 'LOW' if overall < 35 else ('MEDIUM' if overall < 65 else 'HIGH')
    return {
        'overall': overall,
        'level': level,
        'breakdown': {
            'property': round(property_risk, 1),
            'buyer': round(buyer_risk, 1),
            'seller': round(seller_risk, 1),
            'anomaly': round(anomaly_risk, 1),
        },
    }


def default_tx_profile(p, predicted):
    return {
        'escrow_ratio': 0.10,
        'time_between_hours': 168.0,
        'cancel_count': 0.0,
        'failed_count': 0.0,
        'listed_vs_pred_ratio': float(round(p['price_eth'] / predicted, 3)),
        'history_count': 10.0,
        'rapid_activity': 0.0,
    }


def similar_properties(token_id, top_n=3):
    props = [p for p in PROPERTIES if p['token_id'] != token_id]
    target_idx = next(i for i, p in enumerate(PROPERTIES) if p['token_id'] == token_id)
    sims = REC_SIM[target_idx]
    ranked = sorted(
        ((sims[i], p) for i, p in enumerate(PROPERTIES) if p['token_id'] != token_id),
        key=lambda t: t[0],
        reverse=True,
    )[:top_n]
    return [{
        'token_id': p['token_id'],
        'name': p['name'],
        'image': p['image'],
        'price_eth': p['price_eth'],
        'match': round(100 * float(s), 1),
    } for s, p in ranked]


def build_insights(token_id, buyer_risk=10.0, seller_risk=10.0):
    prop = next((p for p in PROPERTIES if p['token_id'] == token_id), None)
    if not prop:
        return None
    predicted = predict_price_eth(prop)
    listed = prop['price_eth']
    diff_pct = round((predicted - listed) / listed * 100, 1) if listed else 0.0
    profile = default_tx_profile(prop, predicted)
    anomaly = anomaly_score(profile)
    risk = compute_risk(predicted, listed, prop['year_built'], anomaly, buyer_risk, seller_risk)
    return {
        'token_id': prop['token_id'],
        'name': prop['name'],
        'image': prop['image'],
        'address': prop['address'],
        'listed_price_eth': listed,
        'predicted_price_eth': predicted,
        'difference_pct': diff_pct,
        'verdict': verdict_for(diff_pct),
        'risk': risk,
        'anomaly_score': anomaly,
        'similar': similar_properties(token_id),
    }


class PriceRequest(BaseModel):
    beds: int
    baths: int
    sqft: float
    year_built: int
    property_type: str
    region: str = 'Other'
    listed_price_eth: float | None = None


class FraudRequest(BaseModel):
    escrow_ratio: float = 0.10
    time_between_hours: float = 168.0
    cancel_count: float = 0.0
    failed_count: float = 0.0
    listed_vs_pred_ratio: float = 1.0
    history_count: float = 10.0
    rapid_activity: float = 0.0


class RiskRequest(BaseModel):
    listed_price_eth: float
    predicted_price_eth: float
    year_built: int
    anomaly_score: float = 0.0
    buyer_risk: float = 10.0
    seller_risk: float = 10.0


@app.get('/')
def root():
    return {
        'service': 'Millow AI Intelligence',
        'status': 'ok',
        'properties': len(PROPERTIES),
        'endpoints': [
            '/api/properties',
            '/predict/price',
            '/predict/fraud',
            '/predict/risk',
            '/recommendations/{token_id}',
            '/property/{token_id}/insights',
            '/market/insights',
        ],
    }


@app.get('/api/properties')
def api_properties():
    return PROPERTIES


@app.post('/predict/price')
def predict_price(req: PriceRequest):
    prop = {
        'beds': req.beds,
        'baths': req.baths,
        'sqft': req.sqft,
        'year_built': req.year_built,
        'property_type': req.property_type,
        'region': req.region,
    }
    predicted = predict_price_eth(prop)
    result = {'predicted_price_eth': predicted}
    if req.listed_price_eth:
        diff = round((predicted - req.listed_price_eth) / req.listed_price_eth * 100, 1)
        result.update({
            'listed_price_eth': req.listed_price_eth,
            'difference_pct': diff,
            'verdict': verdict_for(diff),
        })
    return result


@app.post('/predict/fraud')
def predict_fraud(req: FraudRequest):
    score = anomaly_score(req.model_dump())
    return {'anomaly_score': score, 'flag': 'Suspicious' if score > 0.55 else 'Normal'}


@app.post('/predict/risk')
def predict_risk(req: RiskRequest):
    risk = compute_risk(
        req.predicted_price_eth,
        req.listed_price_eth,
        req.year_built,
        req.anomaly_score,
        req.buyer_risk,
        req.seller_risk,
    )
    return risk


@app.get('/recommendations/{token_id}')
def recommendations(token_id: int):
    return {'token_id': token_id, 'similar': similar_properties(token_id)}


@app.get('/market/insights')
def market_insights():
    rows = [build_insights(p['token_id']) for p in PROPERTIES]
    verdicts = [r['verdict'] for r in rows]
    risks = [r['risk']['overall'] for r in rows]
    stats = {
        'properties_analyzed': len(rows),
        'avg_listed_eth': round(float(np.mean([r['listed_price_eth'] for r in rows])), 2),
        'avg_predicted_eth': round(float(np.mean([r['predicted_price_eth'] for r in rows])), 2),
        'avg_difference_pct': round(float(np.mean([r['difference_pct'] for r in rows])), 1),
        'undervalued': verdicts.count('Undervalued'),
        'overvalued': verdicts.count('Overvalued'),
        'fairly_priced': verdicts.count('Fairly priced'),
        'avg_risk': round(float(np.mean(risks)), 1),
    }
    deals = sorted(rows, key=lambda r: r['difference_pct'], reverse=True)[:3]
    return {
        'stats': stats,
        'deals': [{
            'token_id': d['token_id'],
            'name': d['name'],
            'image': d['image'],
            'listed_price_eth': d['listed_price_eth'],
            'predicted_price_eth': d['predicted_price_eth'],
            'difference_pct': d['difference_pct'],
            'verdict': d['verdict'],
        } for d in deals],
    }


@app.get('/property/{token_id}/insights')
def property_insights(token_id: int):
    insights = build_insights(token_id)
    if not insights:
        return {'error': f'property {token_id} not found'}
    return insights


if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8000)
