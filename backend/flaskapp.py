from flask import Flask, request, jsonify
from flask_cors import CORS
import backtrader as bt
import yfinance as yf
import pandas as pd
from datetime import datetime
import numpy as np
from scipy.stats import linregress
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
import json
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# --- UTILITY TO CONVERT DATES IN NESTED OBJECTS ---
def convert_dates_to_iso(obj):
    """Recursively convert datetime.date objects to ISO 8601 strings."""
    if isinstance(obj, dict):
        return {k: convert_dates_to_iso(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_dates_to_iso(i) for i in obj]
    elif isinstance(obj, datetime):
        return obj.date().isoformat()
    elif hasattr(obj, 'isoformat'):
        return obj.isoformat()
    return obj

# --- ENHANCED STRATEGY DEFINITIONS WITH SIGNAL TRACKING ---

class CrossSectionalMeanReversion(bt.Strategy):
    def __init__(self):
        self.stock_data = self.datas
        self.portfolio_values = []
        self.dates = []
        self.signals = []
        self.price_data = {}
        self.indicators = {} # No specific indicators for this strategy, but keep for consistency

        # Initialize price tracking for each stock
        for stock in self.stock_data:
            self.price_data[stock._name] = []

    def prenext(self):
        self.next()

    def next(self):
        current_date = self.datas[0].datetime.date(0)
        self.portfolio_values.append(self.broker.getvalue())
        self.dates.append(current_date)

        # Store price data for each stock
        for stock in self.stock_data:
            self.price_data[stock._name].append({
                'date': current_date,
                'price': stock.close[0]
            })

        stock_returns = np.zeros(len(self.stock_data))
        for index, stock in enumerate(self.stock_data):
            if len(stock.close) > 1:
                 stock_returns[index] = (stock.close[0] - stock.close[-1]) / stock.close[-1]
            else:
                 stock_returns[index] = 0


        market_return = np.mean(stock_returns)
        weights = -(stock_returns - market_return)
        # Normalize weights
        weight_sum = np.sum(np.abs(weights))
        if weight_sum > 0:
            weights = weights / weight_sum
        else:
            weights = np.zeros(len(self.stock_data))


        for index, stock in enumerate(self.stock_data):
            target_value = self.broker.getvalue() * weights[index]
            
            # Record significant trades as signals
            if abs(self.getposition(stock).size * stock.close[0] - target_value) > 0.01 * self.broker.getvalue():
                signal_type = 'BUY' if target_value > self.getposition(stock).size * stock.close[0] else 'SELL'
                self.signals.append({
                    'date': current_date,
                    'ticker': stock._name,
                    'type': signal_type,
                    'price': stock.close[0],
                    'weight': weights[index]
                })
            
            self.order_target_percent(stock, target=weights[index])

class Momentum(bt.Indicator):
    lines = ('momentum_trend',)
    params = (('period', 90),)

    def __init__(self):
        self.addminperiod(self.params.period)

    def next(self):
        returns = np.log(self.data.get(size=self.params.period))
        x = np.arange(len(returns))
        # Check for sufficient data points to avoid errors
        if len(x) < 2:
            self.lines.momentum_trend[0] = 0
            return
        beta, _, rvalue, _, _ = linregress(x, returns)
        annualized = (1 + beta) ** 252
        self.lines.momentum_trend[0] = annualized * (rvalue ** 2)

class MomentumPortfolioStrategy(bt.Strategy):
    params = (
        ('momentum_period', 90),
        ('rebalance_days', 5),
        ('num_top_stocks', 0.2),
    )

    def __init__(self):
        self.counter = 0
        self.portfolio_values = []
        self.dates = []
        self.signals = []
        self.price_data = {}
        
        # This will store the indicator objects, keyed by stock name
        self.momentum_indicators = {}
        for stock in self.datas:
            self.momentum_indicators[stock._name] = Momentum(stock, period=self.p.momentum_period)
            self.price_data[stock._name] = []
            
        # This will store the calculated indicator values, keyed by date
        self.indicators = {}

    def next(self):
        current_date = self.datas[0].datetime.date(0)
        self.portfolio_values.append(self.broker.getvalue())
        self.dates.append(current_date)
        
        # Store price data and indicator data for each stock
        current_indicators = {}
        for stock in self.datas:
            self.price_data[stock._name].append({
                'date': current_date,
                'price': stock.close[0]
            })
            # Get the momentum value from the indicator object
            momentum_value = self.momentum_indicators[stock._name][0]
            if not np.isnan(momentum_value):
                current_indicators[f'momentum_{stock._name}'] = momentum_value
        
        if current_indicators:
            self.indicators[current_date] = current_indicators
            
        if self.counter % self.p.rebalance_days == 0:
            self.rebalance_portfolio()
        self.counter += 1

    def rebalance_portfolio(self):
        # Get current momentum values for all stocks
        momentum_values = {stock: self.momentum_indicators[stock._name][0] for stock in self.datas}

        # Filter out stocks with NaN momentum values
        valid_stocks = {s: m for s, m in momentum_values.items() if not np.isnan(m)}
        if not valid_stocks:
            return

        # Sort stocks by momentum
        sorted_stocks = sorted(valid_stocks.keys(), key=lambda s: valid_stocks[s], reverse=True)
        
        num_stocks_to_trade = int(np.ceil(len(self.datas) * self.p.num_top_stocks))
        top_stocks = sorted_stocks[:num_stocks_to_trade]
        
        current_date = self.datas[0].datetime.date(0)

        for stock in self.datas:
            if stock in top_stocks:
                if not self.getposition(stock):
                    self.signals.append({
                        'date': current_date,
                        'ticker': stock._name,
                        'type': 'BUY',
                        'price': stock.close[0],
                        'momentum': valid_stocks.get(stock)
                    })
                self.order_target_percent(stock, target=1.0 / num_stocks_to_trade)
            else:
                if self.getposition(stock):
                    self.signals.append({
                        'date': current_date,
                        'ticker': stock._name,
                        'type': 'SELL',
                        'price': stock.close[0],
                        'momentum': valid_stocks.get(stock)
                    })
                self.order_target_percent(stock, target=0.0)

class PairsTradingStrategy(bt.Strategy):
    params = (
        ('lookback', 60),
        ('entry_threshold', 2.0),
        ('exit_threshold', 0.5),
        ('trade_size_percent', 0.1)
    )

    def __init__(self):
        self.stock1 = self.datas[0]
        self.stock2 = self.datas[1]
        self.portfolio_values = []
        self.dates = []
        self.signals = []
        self.price_data = {self.stock1._name: [], self.stock2._name: []}
        self.indicators = {}  # Store spread and zscore data

    def next(self):
        current_date = self.datas[0].datetime.date(0)
        self.portfolio_values.append(self.broker.getvalue())
        self.dates.append(current_date)
        
        self.price_data[self.stock1._name].append({'date': current_date, 'price': self.stock1.close[0]})
        self.price_data[self.stock2._name].append({'date': current_date, 'price': self.stock2.close[0]})
        
        prices1 = pd.Series(self.stock1.close.get(size=self.p.lookback))
        prices2 = pd.Series(self.stock2.close.get(size=self.p.lookback))

        if len(prices1) < self.p.lookback or len(prices2) < self.p.lookback:
            return

        try:
            prices1_const = sm.add_constant(prices1)
            model = sm.OLS(prices2, prices1_const).fit()
            hedge_ratio = model.params[1]
            
            spread = prices2 - hedge_ratio * prices1
            zscore = (spread.iloc[-1] - spread.mean()) / spread.std()
            
            self.indicators[current_date] = {
                'spread': spread.iloc[-1],
                'zscore': zscore,
                'hedge_ratio': hedge_ratio
            }

            if not self.position:
                if zscore > self.p.entry_threshold:
                    self.buy(data=self.stock1, size=(self.broker.getvalue() * self.p.trade_size_percent) / self.stock1.close[0])
                    self.sell(data=self.stock2, size=(self.broker.getvalue() * self.p.trade_size_percent * hedge_ratio) / self.stock2.close[0])
                    self.signals.append({'date': current_date, 'ticker': self.stock1._name, 'type': 'BUY', 'price': self.stock1.close[0], 'zscore': zscore})
                    self.signals.append({'date': current_date, 'ticker': self.stock2._name, 'type': 'SELL', 'price': self.stock2.close[0], 'zscore': zscore})
                elif zscore < -self.p.entry_threshold:
                    self.sell(data=self.stock1, size=(self.broker.getvalue() * self.p.trade_size_percent) / self.stock1.close[0])
                    self.buy(data=self.stock2, size=(self.broker.getvalue() * self.p.trade_size_percent * hedge_ratio) / self.stock2.close[0])
                    self.signals.append({'date': current_date, 'ticker': self.stock1._name, 'type': 'SELL', 'price': self.stock1.close[0], 'zscore': zscore})
                    self.signals.append({'date': current_date, 'ticker': self.stock2._name, 'type': 'BUY', 'price': self.stock2.close[0], 'zscore': zscore})
            elif abs(zscore) < self.p.exit_threshold:
                self.close(self.stock1)
                self.close(self.stock2)
                self.signals.append({'date': current_date, 'ticker': self.stock1._name, 'type': 'CLOSE', 'price': self.stock1.close[0], 'zscore': zscore})
                self.signals.append({'date': current_date, 'ticker': self.stock2._name, 'type': 'CLOSE', 'price': self.stock2.close[0], 'zscore': zscore})
        except Exception as e:
            # print(f"Error in Pairs Trading: {e}")
            pass

class RollingLogisticRegressionStrategy(bt.Strategy):
    params = (('lookback', 252), ('rebalance_days', 21),)

    def __init__(self):
        self.models = {}
        self.counter = 0
        self.portfolio_values = []
        self.dates = []
        self.signals = []
        self.price_data = {}
        self.indicators = {} # Store prediction confidence

        for stock in self.datas:
            self.price_data[stock._name] = []

    def next(self):
        current_date = self.datas[0].datetime.date(0)
        self.portfolio_values.append(self.broker.getvalue())
        self.dates.append(current_date)
        
        for stock in self.datas:
            self.price_data[stock._name].append({'date': current_date, 'price': stock.close[0]})
            
        if self.counter % self.p.rebalance_days == 0:
            self.train_models()
        self.execute_trades()
        self.counter += 1

    def train_models(self):
        for stock in self.datas:
            try:
                prices = pd.Series(stock.close.get(size=self.p.lookback + 1))
                if len(prices) < self.p.lookback + 1: continue
                
                returns = np.log(prices / prices.shift(1)).dropna()
                df = pd.DataFrame({'returns': returns})
                df['direction'] = np.where(df['returns'] > 0, 1, 0)
                df['lag1'] = df['returns'].shift(1)
                df.dropna(inplace=True)

                if len(df) < 20: continue

                X = df[['lag1']]
                y = df['direction']
                model = LogisticRegression()
                model.fit(X, y)
                self.models[stock] = model
            except Exception:
                continue

    def execute_trades(self):
        current_date = self.datas[0].datetime.date(0)
        for stock in self.datas:
            if stock in self.models:
                try:
                    feature = pd.DataFrame([np.log(stock.close[0] / stock.close[-1])], columns=['lag1'])
                    prediction = self.models[stock].predict(feature)[0]
                    prediction_proba = self.models[stock].predict_proba(feature)[0]
                    self.indicators[current_date] = self.indicators.get(current_date, {})
                    self.indicators[current_date][f'{stock._name}_confidence'] = prediction_proba[prediction]

                    if prediction == 1 and not self.getposition(stock):
                        self.order_target_percent(stock, target=1.0 / len(self.datas))
                        self.signals.append({'date': current_date, 'ticker': stock._name, 'type': 'BUY', 'price': stock.close[0], 'confidence': prediction_proba[1]})
                    elif prediction == 0 and self.getposition(stock):
                        self.order_target_percent(stock, target=0.0)
                        self.signals.append({'date': current_date, 'ticker': stock._name, 'type': 'SELL', 'price': stock.close[0], 'confidence': prediction_proba[0]})
                except Exception:
                    pass

class EMACrossoverStrategy(bt.Strategy):
    params = (('short_period', 30), ('long_period', 100),)

    def __init__(self):
        self.dataclose = self.datas[0].close
        self.order = None
        self.portfolio_values, self.dates, self.signals = [], [], []
        self.price_data = {self.datas[0]._name: []}
        self.short_ma = bt.indicators.EMA(self.datas[0], period=self.p.short_period)
        self.long_ma = bt.indicators.EMA(self.datas[0], period=self.p.long_period)
        self.crossover = bt.indicators.CrossOver(self.short_ma, self.long_ma)
        self.indicators = {}

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]: return
        self.order = None

    def next(self):
        current_date = self.datas[0].datetime.date(0)
        self.portfolio_values.append(self.broker.getvalue())
        self.dates.append(current_date)
        
        self.price_data[self.datas[0]._name].append({'date': current_date, 'price': self.dataclose[0]})
        self.indicators[current_date] = {'short_ma': self.short_ma[0], 'long_ma': self.long_ma[0]}
        
        if self.order: return

        if not self.position and self.crossover > 0:
            self.order = self.buy()
            self.signals.append({'date': current_date, 'ticker': self.datas[0]._name, 'type': 'BUY', 'price': self.dataclose[0]})
        elif self.position and self.crossover < 0:
            self.order = self.sell()
            self.signals.append({'date': current_date, 'ticker': self.datas[0]._name, 'type': 'SELL', 'price': self.dataclose[0]})

class SMACrossoverStrategy(bt.Strategy):
    params = (('short_period', 30), ('long_period', 100),)

    def __init__(self):
        self.dataclose = self.datas[0].close
        self.order = None
        self.portfolio_values, self.dates, self.signals = [], [], []
        self.price_data = {self.datas[0]._name: []}
        self.short_ma = bt.indicators.SMA(self.datas[0], period=self.p.short_period)
        self.long_ma = bt.indicators.SMA(self.datas[0], period=self.p.long_period)
        self.crossover = bt.indicators.CrossOver(self.short_ma, self.long_ma)
        self.indicators = {}

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]: return
        self.order = None

    def next(self):
        current_date = self.datas[0].datetime.date(0)
        self.portfolio_values.append(self.broker.getvalue())
        self.dates.append(current_date)
        
        self.price_data[self.datas[0]._name].append({'date': current_date, 'price': self.dataclose[0]})
        self.indicators[current_date] = {'short_ma': self.short_ma[0], 'long_ma': self.long_ma[0]}
        
        if self.order: return

        if not self.position and self.crossover > 0:
            self.order = self.buy()
            self.signals.append({'date': current_date, 'ticker': self.datas[0]._name, 'type': 'BUY', 'price': self.dataclose[0]})
        elif self.position and self.crossover < 0:
            self.order = self.sell()
            self.signals.append({'date': current_date, 'ticker': self.datas[0]._name, 'type': 'SELL', 'price': self.dataclose[0]})

class BollingerBandStrategy(bt.Strategy):
    params = (('period', 20), ('std', 2),)

    def __init__(self):
        self.bollinger = bt.indicators.BollingerBands(period=self.p.period, devfactor=self.p.std)
        self.order = None
        self.portfolio_values, self.dates, self.signals = [], [], []
        self.price_data = {self.datas[0]._name: []}
        self.indicators = {}

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]: return
        self.order = None

    def next(self):
        current_date = self.datas[0].datetime.date(0)
        self.portfolio_values.append(self.broker.getvalue())
        self.dates.append(current_date)
        
        self.price_data[self.datas[0]._name].append({'date': current_date, 'price': self.data.close[0]})
        self.indicators[current_date] = {
            'upper_band': self.bollinger.lines.top[0],
            'middle_band': self.bollinger.lines.mid[0],
            'lower_band': self.bollinger.lines.bot[0]
        }
        
        if self.order: return

        if not self.position:
            if self.data.close[0] < self.bollinger.lines.bot[0]:
                self.order = self.buy()
                self.signals.append({'date': current_date, 'ticker': self.datas[0]._name, 'type': 'BUY', 'price': self.data.close[0]})
        elif self.data.close[0] > self.bollinger.lines.mid[0]:
            self.order = self.close()
            self.signals.append({'date': current_date, 'ticker': self.datas[0]._name, 'type': 'CLOSE', 'price': self.data.close[0]})

class RsiMaStrategy(bt.Strategy):
    params = (('short_period', 40), ('long_period', 150), ('rsi_period', 14), ('rsi_oversold', 30),)

    def __init__(self):
        self.dataclose = self.datas[0].close
        self.order = None
        self.portfolio_values, self.dates, self.signals = [], [], []
        self.price_data = {self.datas[0]._name: []}
        self.short_ma = bt.indicators.EMA(self.datas[0], period=self.p.short_period)
        self.long_ma = bt.indicators.EMA(self.datas[0], period=self.p.long_period)
        self.rsi = bt.indicators.RSI(self.datas[0], period=self.p.rsi_period)
        self.indicators = {}

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]: return
        self.order = None

    def next(self):
        current_date = self.datas[0].datetime.date(0)
        self.portfolio_values.append(self.broker.getvalue())
        self.dates.append(current_date)
        
        self.price_data[self.datas[0]._name].append({'date': current_date, 'price': self.dataclose[0]})
        self.indicators[current_date] = {'short_ma': self.short_ma[0], 'long_ma': self.long_ma[0], 'rsi': self.rsi[0]}
        
        if self.order: return

        if not self.position and self.short_ma > self.long_ma and self.rsi < self.p.rsi_oversold:
            self.order = self.buy()
            self.signals.append({'date': current_date, 'ticker': self.datas[0]._name, 'type': 'BUY', 'price': self.dataclose[0]})
        elif self.position and self.short_ma < self.long_ma:
            self.order = self.sell()
            self.signals.append({'date': current_date, 'ticker': self.datas[0]._name, 'type': 'SELL', 'price': self.dataclose[0]})

class MomentumSMACrossoverStrategy(bt.Strategy):
    params = (('momentum_period', 90), ('sma_period', 200),)

    def __init__(self):
        self.dataclose = self.datas[0].close
        self.order = None
        self.portfolio_values, self.dates, self.signals = [], [], []
        self.price_data = {self.datas[0]._name: []}
        self.momentum = Momentum(self.datas[0], period=self.p.momentum_period)
        self.sma = bt.indicators.SMA(self.datas[0], period=self.p.sma_period)
        self.indicators = {}

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]: return
        self.order = None

    def next(self):
        current_date = self.datas[0].datetime.date(0)
        self.portfolio_values.append(self.broker.getvalue())
        self.dates.append(current_date)
        
        self.price_data[self.datas[0]._name].append({'date': current_date, 'price': self.dataclose[0]})
        self.indicators[current_date] = {'momentum': self.momentum[0], 'sma': self.sma[0]}
        
        if self.order: return

        if not self.position and self.momentum > 1.0 and self.dataclose[0] > self.sma[0]:
            self.order = self.buy()
            self.signals.append({'date': current_date, 'ticker': self.datas[0]._name, 'type': 'BUY', 'price': self.dataclose[0]})
        elif self.position and self.dataclose[0] < self.sma[0]:
            self.order = self.sell()
            self.signals.append({'date': current_date, 'ticker': self.datas[0]._name, 'type': 'SELL', 'price': self.dataclose[0]})


# --- ENHANCED BACKTESTING ENGINE ---
def run_backtest_api(strategy_class, tickers, from_date, to_date, cash=100000, commission=0.001):
    cerebro = bt.Cerebro()
    cerebro.addstrategy(strategy_class)

    for ticker in tickers:
        try:
            df = yf.download(ticker, start=from_date, end=to_date, group_by='ticker')
            
            if df.empty:
                print(f"No data for {ticker}, skipping.")
                continue

            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.droplevel(0)
            
            df.columns = [col.lower() for col in df.columns]

            data = bt.feeds.PandasData(dataname=df, name=ticker)
            cerebro.adddata(data, name=ticker)
        except Exception as e:
            print(f"Error downloading or processing data for {ticker}: {e}")
            continue
    
    if not cerebro.datas:
        raise ValueError("No valid data loaded for any ticker.")

    cerebro.broker.set_cash(cash)
    cerebro.broker.setcommission(commission=commission)
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe_ratio', timeframe=bt.TimeFrame.Days, riskfreerate=0.0)
    cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')

    initial_value = cerebro.broker.getvalue()
    results = cerebro.run()
    final_value = cerebro.broker.getvalue()
    strategy_instance = results[0]

    # --- DATA EXTRACTION AND PROCESSING ---
    portfolio_values = strategy_instance.portfolio_values
    dates = strategy_instance.dates
    signals = getattr(strategy_instance, 'signals', [])
    price_data_raw = getattr(strategy_instance, 'price_data', {})
    indicators_raw = getattr(strategy_instance, 'indicators', {})

    peak = initial_value
    drawdown_data = []
    for value in portfolio_values:
        if value > peak: peak = value
        drawdown = (peak - value) / peak * 100 if peak > 0 else 0
        drawdown_data.append(drawdown)

    chart_data = [{
        'date': date.isoformat(),
        'portfolioValue': round(value, 2),
        'drawdown': round(dd, 2)
    } for date, value, dd in zip(dates, portfolio_values, drawdown_data)]

    price_dfs = []
    for ticker, data in price_data_raw.items():
        if data:
            df = pd.DataFrame(data)
            df['date'] = pd.to_datetime(df['date'])
            df = df.set_index('date')
            df.rename(columns={'price': f'price_{ticker}'}, inplace=True)
            price_dfs.append(df)
    
    indicators_df = pd.DataFrame.from_dict(indicators_raw, orient='index')
    if not indicators_df.empty:
        indicators_df.index = pd.to_datetime(indicators_df.index)

    if price_dfs:
        valuation_df = pd.concat(price_dfs, axis=1)
        if not indicators_df.empty:
            valuation_df = valuation_df.join(indicators_df)
        valuation_df.reset_index(inplace=True)
        valuation_data = valuation_df.replace({np.nan: None}).to_dict('records')
        valuation_data = convert_dates_to_iso(valuation_data)
    else:
        valuation_data = []

    analyzers = strategy_instance.analyzers
    sharpe = analyzers.sharpe_ratio.get_analysis().get('sharperatio', 0)
    returns = analyzers.returns.get_analysis().get('rnorm100', 0)
    max_drawdown = analyzers.drawdown.get_analysis().get('max', {}).get('drawdown', 0)
    
    total_return = ((final_value - initial_value) / initial_value) * 100 if initial_value > 0 else 0
    
    metrics = {
        'totalReturn': round(total_return, 2),
        'annualizedReturn': round(returns, 2),
        'sharpeRatio': round(sharpe, 2) if sharpe is not None else 0,
        'maxDrawdown': round(max_drawdown, 2),
        'finalValue': round(final_value, 2),
        'initialValue': round(initial_value, 2)
    }
    
    return {
        'data': chart_data,
        'metrics': metrics,
        'tickers': tickers,
        'valuationData': valuation_data,
        'signals': convert_dates_to_iso(signals),
        'status': 'success'
    }

# --- API ROUTES ---
@app.route('/api/strategies', methods=['GET'])
def get_strategies():
    strategies = {
        '1': {'name': 'EMA Crossover', 'type': 'single', 'tickers': ['MSFT']},
        '2': {'name': 'SMA Crossover', 'type': 'single', 'tickers': ['AAPL']},
        '3': {'name': 'Bollinger Bands', 'type': 'single', 'tickers': ['GOOGL']},
        '4': {'name': 'RSI + MA Crossover', 'type': 'single', 'tickers': ['AMZN']},
        '5': {'name': 'Momentum + SMA', 'type': 'single', 'tickers': ['TSLA']},
        '6': {'name': 'Cross-Sectional Mean Reversion', 'type': 'multi', 'tickers': ['AAPL', 'MSFT', 'GOOGL', 'AMZN']},
        '7': {'name': 'Momentum Portfolio', 'type': 'multi', 'tickers': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META']},
        '8': {'name': 'Pairs Trading', 'type': 'multi', 'tickers': ['XOM', 'CVX']},
        '9': {'name': 'Rolling Logistic Regression', 'type': 'multi', 'tickers': ['AAPL', 'MSFT', 'GOOGL']}
    }
    return jsonify(strategies)

@app.route('/api/backtest', methods=['POST'])
def run_backtest():
    try:
        data = request.json
        strategy_id = data.get('strategy_id')
        tickers_str = data.get('tickers', [])
        start_date_str = data.get('start_date', '2018-01-01')
        end_date_str = data.get('end_date', '2024-01-01')
        
        start_date = datetime.strptime(start_date_str, '%Y-%m-%d')
        end_date = datetime.strptime(end_date_str, '%Y-%m-%d')
        
        strategy_map = {
            '1': EMACrossoverStrategy, '2': SMACrossoverStrategy, '3': BollingerBandStrategy,
            '4': RsiMaStrategy, '5': MomentumSMACrossoverStrategy, '6': CrossSectionalMeanReversion,
            '7': MomentumPortfolioStrategy, '8': PairsTradingStrategy, '9': RollingLogisticRegressionStrategy
        }
        
        if strategy_id not in strategy_map:
            return jsonify({'error': 'Invalid strategy ID'}), 400
        
        strategy_class = strategy_map[strategy_id]
        
        result = run_backtest_api(strategy_class, tickers_str, start_date, end_date)
        return jsonify(result)
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e), 'status': 'error'}), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy', 'message': 'Backtest API is running'})

if __name__ == '__main__':
    print("Starting Flask Backtest API...")
    app.run(debug=True, host='0.0.0.0', port=5000)
