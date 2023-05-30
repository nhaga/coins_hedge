import datetime

import ccxt
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from openbb_terminal.sdk import openbb

binance = ccxt.binance()


symbols = [
    'XRP',
    'MATIC',
    'ADA',
    'SOL',
    'LTC',
    'DOT',
    #'AVAX',
    'LINK',
    'UNI',
    'XMR',
    'XLM'
]


st.sidebar.title("3CommaDigital")
st.sidebar.header("Hedge Portfolio")


@st.cache_data
def fetch_coins():
    return openbb.crypto.disc.top_coins(limit=50).sort_values('market_cap', ascending=False)

coins = fetch_coins()

flds = ['id', 'symbol', 'market_cap']
st.sidebar.dataframe(
    coins[coins.symbol.str.upper().isin(symbols)][flds].set_index('symbol')
) 

method = st.sidebar.radio(
    "Weight Methodology",
    (
        'Equal Weights',
        'Market Cap',
        'Mean Variance Optimization',
        'Hierarchical Risk Parity (HRP)',
        'Mean Conditional Value at Risk (mCVAR)',
        'Conditional drawdown at Risk (CDaR)'
        ))


start = st.sidebar.date_input(
    "Select start date",
    datetime.date(2022, 1, 1))

st.write('Analysis start date:', start)
start = datetime.datetime.combine(start, datetime.datetime.min.time())



@st.cache_data
def get_portfolio(start):
    tickers = [f"{symbol}/USDT" for symbol in symbols]
    flds = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
    pxs = []
    for ticker in tickers:
        data = binance.fetch_ohlcv(ticker, '1d', since=int(start.timestamp()*1000))
        px = pd.DataFrame(data, columns=flds).set_index('timestamp')        
        df = px[['close']].rename(columns={"close": ticker})
        pxs.append(df)    
    pxs = pd.concat(pxs, axis=1)    
    pxs.index = pxs.index.map(lambda x: datetime.datetime.fromtimestamp(x/1000))
    return pxs


@st.cache_data
def get_benchmark(start):
    eth = binance.fetch_ohlcv("ETH/USDT", '1d', since=int(start.timestamp()*1000))
    flds = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
    eth = pd.DataFrame(eth, columns=flds).set_index('timestamp')
    eth.index = eth.index.map(lambda x: datetime.datetime.fromtimestamp(x/1000))
    eth = eth[['close']].rename(columns={"close": "ETH"})
    return eth


portfolio = get_portfolio(start)
eth = get_benchmark(start)


def make_portfolio(tickers, weights, name='Hedge Portfolio'):
    symbols = list(map(lambda x: x.split('/')[0].lower(), tickers))
    res = [{"symbol": symbol, "weight": w } for symbol, w in zip(symbols, weights)]
    prices = portfolio[tickers]
    p = (prices @ weights)
    p.name = name
    return p, res


if method in ('Equal Weights', 'Market Cap'):
    selected_symbols = st.sidebar.multiselect(
        'Select coins for Hedge',
        symbols,
        ['XRP', 'MATIC'])

    selected_tickers = [f"{symbol}/USDT" for symbol in selected_symbols]


    if method == 'Equal Weights':
        N = len(selected_symbols)
        weights = np.array([1 / N] * N)

    if method == 'Market Cap':
        market_caps = coins[coins.symbol.str.upper().isin(selected_symbols)].market_cap
        weights = (market_caps  / market_caps.sum()).values



    p, res = make_portfolio(selected_tickers, weights)

from pypfopt import HRPOpt
from pypfopt.efficient_frontier import (EfficientCDaR, EfficientCVaR,
                                        EfficientFrontier)
from pypfopt.expected_returns import mean_historical_return
from pypfopt.risk_models import CovarianceShrinkage

if method == "Mean Variance Optimization":
    mu = mean_historical_return(portfolio)
    S = CovarianceShrinkage(portfolio).ledoit_wolf()

    ef = EfficientFrontier(mu, S)
    weights = ef.min_volatility()

    cleaned_weights = ef.clean_weights()

    tickers = [ticker for ticker, weight in cleaned_weights.items() if weight > 0]
    weights = np.array([weight for ticker, weight in cleaned_weights.items() if weight > 0])
    p, res = make_portfolio(tickers, weights)

if method == 'Hierarchical Risk Parity (HRP)':
    returns = portfolio.pct_change().dropna()
    hrp = HRPOpt(returns)
    cleaned_weights = hrp.optimize()

    tickers = [ticker for ticker, weight in cleaned_weights.items() if weight > 0]
    weights = np.array([weight for ticker, weight in cleaned_weights.items() if weight > 0])
    p, res = make_portfolio(tickers, weights)


if method == 'Mean Conditional Value at Risk (mCVAR)':
    S = portfolio.cov()
    mu = mean_historical_return(portfolio)
    ef_cvar = EfficientCVaR(mu, S)
    cvar_weights = ef_cvar.min_cvar()

    cleaned_weights = ef_cvar.clean_weights()
    tickers = [ticker for ticker, weight in cleaned_weights.items() if weight > 0]
    weights = np.array([weight for ticker, weight in cleaned_weights.items() if weight > 0])
    p, res = make_portfolio(tickers, weights)


if method == 'Conditional drawdown at Risk (CDaR)':
    S = portfolio.cov()
    mu = mean_historical_return(portfolio)
    ef_cdar = EfficientCDaR(mu, S)
    cdar_weights = ef_cdar.min_cdar()

    cleaned_weights = ef_cdar.clean_weights()

    tickers = [ticker for ticker, weight in cleaned_weights.items() if weight > 0]
    weights = np.array([weight for ticker, weight in cleaned_weights.items() if weight > 0])
    p, res = make_portfolio(tickers, weights)


st.subheader("Weights")
col1, col2 = st.columns(2)
col1.dataframe(pd.DataFrame(res).style.format({"weight": "{:.2%}"}))
col2.dataframe(eth.join(p).corr())

st.subheader("Cumulative Returns")
st.line_chart(eth.join(p).pct_change().cumsum())

st.subheader("Scatter")
fig = px.scatter(
    x=eth['ETH'].pct_change().dropna().values, 
    y=p.pct_change().dropna().values
    )

st.plotly_chart(fig, use_container_width=True)
