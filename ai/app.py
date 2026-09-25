import json
import os

import numpy as np
import joblib
import uvicorn

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

import train
from chat import ChatAgent
from mreid_tools import (
    MREID_HANDLERS,
    MREID_SYSTEM_PROMPT,
    MREID_TOOLS,
)
import mreid_tools

BASE_DIR = os.path.dirname(__file__)
MODEL_DIR = os.path.join(BASE_DIR, 'models')
PROPS_CACHE = os.path.join(BASE_DIR, 'properties.json')

load_dotenv(os.path.join(BASE_DIR, '.env'))

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


FORECAST_ANNUAL_APPRECIATION = 0.04


def forecast_price(token_id, years=5):
    prop = next((p for p in PROPERTIES if p['token_id'] == token_id), None)
    if not prop:
        return {'error': f'property {token_id} not found'}
    # Sensible bounds so a model cannot produce a nonsense horizon
    years = max(1, min(int(years), 30))
    fair = predict_price_eth(prop)
    projected = round(fair * (1 + FORECAST_ANNUAL_APPRECIATION) ** years, 2)
    return {
        'token_id': token_id,
        'name': prop['name'],
        'current_predicted_eth': fair,
        'projected_price_eth': projected,
        'years': years,
        'annual_appreciation': FORECAST_ANNUAL_APPRECIATION,
        'note': f'Projection assumes a flat {FORECAST_ANNUAL_APPRECIATION:.0%} annual '
                'appreciation in fair value; actual markets vary.',
    }


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
    if not any(p['token_id'] == token_id for p in PROPERTIES):
        return []
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


AGENT_TOOLS = [
    {
        'type': 'function',
        'function': {
            'name': 'predict_price',
            'description': (
                'Predict the fair market price (in ETH) of a property from its features, '
                'and optionally compare it against a listing price.'
            ),
            'parameters': {
                'type': 'object',
                'properties': {
                    'beds': {'type': 'integer', 'description': 'Number of bedrooms'},
                    'baths': {'type': 'integer', 'description': 'Number of bathrooms'},
                    'sqft': {'type': 'number', 'description': 'Square footage'},
                    'year_built': {'type': 'integer', 'description': 'Year the property was built'},
                    'property_type': {'type': 'string', 'enum': train.PROPERTY_TYPES, 'description': 'Type of residence'},
                    'region': {'type': 'string', 'description': 'State or region (US state code or Other)'},
                    'listed_price_eth': {'type': 'number', 'description': 'Optional listing price to compare against'},
                },
                'required': ['beds', 'baths', 'sqft', 'year_built', 'property_type'],
            },
        },
    },
    {
        'type': 'function',
        'function': {
            'name': 'get_property_insights',
            'description': (
                'Full AI analysis for a property on Millow by its numeric token id: predicted vs '
                'listed price, valuation verdict, risk level, anomaly score and similar listings.'
            ),
            'parameters': {
                'type': 'object',
                'properties': {
                    'token_id': {'type': 'integer', 'description': 'Token id of the property (1-24)'},
                },
                'required': ['token_id'],
            },
        },
    },
    {
        'type': 'function',
        'function': {
            'name': 'get_market_insights',
            'description': (
                'Overall market statistics for all Millow listings: average listed and predicted '
                'price, valuation verdict counts, average risk, and the best (undervalued) deals.'
            ),
            'parameters': {'type': 'object', 'properties': {}},
        },
    },
    {
        'type': 'function',
        'function': {
            'name': 'get_similar_properties',
            'description': 'Find the most similar properties to a given token id.',
            'parameters': {
                'type': 'object',
                'properties': {
                    'token_id': {'type': 'integer', 'description': 'Token id of the property (1-24)'},
                    'top_n': {'type': 'integer', 'description': 'Optional number of results (default 3)'},
                },
                'required': ['token_id'],
            },
        },
    },
    {
        'type': 'function',
        'function': {
            'name': 'forecast_price',
            'description': (
                'Estimate how much a property will be worth in the future (default 5 years). '
                'Use this whenever the user asks about future prices, forecasts, appreciation, '
                'or "how much will this be worth in N years".'
            ),
            'parameters': {
                'type': 'object',
                'properties': {
                    'token_id': {'type': 'integer', 'description': 'Token id of the property (1-24)'},
                    'years': {'type': 'integer', 'description': 'Number of years ahead to forecast (default 5)'},
                },
                'required': ['token_id'],
            },
        },
    },
    {
        'type': 'function',
        'function': {
            'name': 'check_transaction_fraud',
            'description': (
                'Score a sale transaction for fraud/anomaly risk based on escrow ratio, timing, '
                'history, cancellations and failed payments.'
            ),
            'parameters': {
                'type': 'object',
                'properties': {
                    'escrow_ratio': {'type': 'number', 'description': 'Earnest money deposit as a fraction of price'},
                    'listed_vs_pred_ratio': {'type': 'number', 'description': 'Listed price divided by predicted price'},
                    'time_between_hours': {'type': 'number', 'description': 'Hours between listing and purchase'},
                    'history_count': {'type': 'number', 'description': 'Number of past transactions'},
                },
            },
        },
    },
]


def handle_predict_price(**kwargs):
    kwargs.pop('top_n', None)
    predicted = predict_price_eth({k: kwargs.get(k) for k in ('beds', 'baths', 'sqft', 'year_built', 'property_type', 'region')})
    body = {'predicted_price_eth': predicted}
    listed = kwargs.get('listed_price_eth')
    if listed:
        diff = round((predicted - listed) / listed * 100, 1)
        body.update({'listed_price_eth': listed, 'difference_pct': diff, 'verdict': verdict_for(diff)})
    return body


def handle_property_insights(token_id, top_n=3):
    insights = build_insights(token_id)
    if not insights:
        return {'error': f'property {token_id} not found'}
    return insights


def handle_market_insights():
    return market_insights()


def handle_similar_properties(token_id, top_n=3):
    return {'token_id': token_id, 'similar': similar_properties(token_id, top_n)}


def handle_forecast_price(token_id, years=5):
    return forecast_price(token_id, years)


def handle_fraud_check(**kwargs):
    profile = default_tx_profile(
        {'price_eth': kwargs.get('price_eth', 10.0)},
        kwargs.get('predicted_price_eth', 10.0),
    )
    profile.update({
        'escrow_ratio': kwargs.get('escrow_ratio', profile['escrow_ratio']),
        'listed_vs_pred_ratio': kwargs.get('listed_vs_pred_ratio', profile['listed_vs_pred_ratio']),
        'time_between_hours': kwargs.get('time_between_hours', profile['time_between_hours']),
        'history_count': kwargs.get('history_count', profile['history_count']),
    })
    score = anomaly_score(profile)
    return {'anomaly_score': score, 'flag': 'Suspicious' if score > 0.55 else 'Normal'}


AGENT_HANDLERS = {
    'predict_price': handle_predict_price,
    'get_property_insights': handle_property_insights,
    'get_market_insights': handle_market_insights,
    'get_similar_properties': handle_similar_properties,
    'forecast_price': handle_forecast_price,
    'check_transaction_fraud': handle_fraud_check,
}

AGENT = ChatAgent(tools=AGENT_TOOLS, handlers=AGENT_HANDLERS)

MREID_AGENT = ChatAgent(
    tools=MREID_TOOLS,
    handlers=MREID_HANDLERS,
    system_prompt=MREID_SYSTEM_PROMPT,
)


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: list[ChatMessage]
    provider: str = 'auto'
    context: str | None = None


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
            '/property/{token_id}/forecast',
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


@app.get('/property/{token_id}/forecast')
def property_forecast(token_id: int, years: int = 5):
    return forecast_price(token_id, years)


@app.get('/chat/status')
def chat_status():
    status = AGENT.status()
    status['mreid_agent'] = {
        'tools': [t['function']['name'] for t in MREID_TOOLS],
        'backend': mreid_tools.BACKEND_BASE,
    }
    return status


@app.post('/chat')
def chat(req: ChatRequest):
    messages = [{'role': m.role, 'content': m.content} for m in req.messages]
    return AGENT.chat(messages, provider=req.provider, context=req.context)


@app.post('/chat/mreid')
def chat_mreid(req: ChatRequest):
    """Chat grounded in the REAL MREID (INR) catalogue via backend :8001."""
    messages = [{'role': m.role, 'content': m.content} for m in req.messages]
    return MREID_AGENT.chat(messages, provider=req.provider, context=req.context)


if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8000)
