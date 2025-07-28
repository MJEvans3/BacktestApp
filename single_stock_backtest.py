import backtrader as bt
import yfinance as yf
import pandas as pd
from datetime import datetime
from scipy.stats import linregress
import numpy as np

# --- 1. STRATEGY DEFINITIONS ---

class EMACrossoverStrategy(bt.Strategy):
    """
    An Exponential Moving Average (EMA) crossover strategy.
    - Buy when the short-term EMA crosses above the long-term EMA.
    - Sell when the short-term EMA crosses below the long-term EMA.
    """
    params = (
        ('short_period', 30),
        ('long_period', 100),
    )

    def __init__(self):
        self.dataclose = self.datas[0].close
        self.order = None
        self.short_ma = bt.indicators.ExponentialMovingAverage(
            self.datas[0], period=self.params.short_period
        )
        self.long_ma = bt.indicators.ExponentialMovingAverage(
            self.datas[0], period=self.params.long_period
        )
        self.crossover = bt.indicators.CrossOver(self.short_ma, self.long_ma)

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            return
        self.order = None

    def next(self):
        if self.order:
            return

        if not self.position:
            if self.crossover > 0:
                self.order = self.buy()
        else:
            if self.crossover < 0:
                self.order = self.sell()

class SMACrossoverStrategy(bt.Strategy):
    """
    A Simple Moving Average (SMA) crossover strategy.
    - Buy when the short-term SMA crosses above the long-term SMA.
    - Sell when the short-term SMA crosses below the long-term SMA.
    """
    params = (
        ('short_period', 30),
        ('long_period', 100),
    )

    def __init__(self):
        self.dataclose = self.datas[0].close
        self.order = None
        self.short_ma = bt.indicators.SimpleMovingAverage(
            self.datas[0], period=self.params.short_period
        )
        self.long_ma = bt.indicators.SimpleMovingAverage(
            self.datas[0], period=self.params.long_period
        )
        self.crossover = bt.indicators.CrossOver(self.short_ma, self.long_ma)

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            return
        self.order = None

    def next(self):
        if self.order:
            return

        if not self.position:
            if self.crossover > 0:
                self.order = self.buy()
        else:
            if self.crossover < 0:
                self.order = self.sell()

class BollingerBandStrategy(bt.Strategy):
    """
    A Bollinger Bands mean-reversion strategy.
    - Go short when the price crosses above the top band.
    - Go long when the price crosses below the bottom band.
    - Close position when the price crosses the middle band.
    """
    params = (
        ('period', 20),
        ('std', 2),
    )

    def __init__(self):
        self.bollinger = bt.indicators.BollingerBands(period=self.p.period, devfactor=self.p.std)
        self.order = None
        self.crossover_mid = bt.indicators.CrossOver(self.data.close, self.bollinger.lines.mid)

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            return
        self.order = None

    def next(self):
        if self.order:
            return

        if not self.position:
            if self.data.close[0] > self.bollinger.lines.top:
                self.order = self.sell()
            elif self.data.close[0] < self.bollinger.lines.bot:
                self.order = self.buy()
        else:
            if self.position.size > 0 and self.crossover_mid < 0:
                self.order = self.close()
            elif self.position.size < 0 and self.crossover_mid > 0:
                self.order = self.close()

class RsiMaStrategy(bt.Strategy):
    """
    A strategy combining Moving Average crossover with an RSI filter.
    - Buy when short MA crosses above long MA, but only if RSI is oversold.
    - Sell when short MA crosses below long MA.
    """
    params = (
        ('short_period', 40),
        ('long_period', 150),
        ('rsi_period', 14),
        ('rsi_oversold', 30),
    )

    def __init__(self):
        self.dataclose = self.datas[0].close
        self.order = None
        self.short_ma = bt.indicators.ExponentialMovingAverage(self.datas[0], period=self.params.short_period)
        self.long_ma = bt.indicators.ExponentialMovingAverage(self.datas[0], period=self.params.long_period)
        self.rsi = bt.indicators.RSI(self.datas[0], period=self.params.rsi_period)

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            return
        self.order = None

    def next(self):
        if self.order:
            return

        if not self.position:
            if self.short_ma > self.long_ma and self.rsi < self.params.rsi_oversold:
                self.order = self.buy()
        else:
            if self.short_ma < self.long_ma:
                self.order = self.sell()

class Momentum(bt.Indicator):
    lines = ('momentum_trend',)
    params = (('period', 90),)

    def __init__(self):
        self.addminperiod(self.params.period)

    def next(self):
        returns = np.log(self.data.get(size=self.params.period))
        x = np.arange(len(returns))
        beta, _, rvalue, _, _ = linregress(x, returns)
        annualized = (1 + beta) ** 252
        self.lines.momentum_trend[0] = annualized * rvalue ** 2

class MomentumSMACrossoverStrategy(bt.Strategy):
    """
    A strategy that combines momentum and a Simple Moving Average (SMA).
    - Buy when momentum is positive and the price is above the SMA.
    - Sell when the price crosses below the SMA.
    """
    params = (
        ('momentum_period', 90),
        ('sma_period', 200),
    )

    def __init__(self):
        self.dataclose = self.datas[0].close
        self.order = None
        self.momentum = Momentum(self.datas[0], period=self.params.momentum_period)
        self.sma = bt.indicators.SimpleMovingAverage(
            self.datas[0], period=self.params.sma_period
        )

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            return
        self.order = None

    def next(self):
        if self.order:
            return

        if not self.position:
            if self.momentum > 1.0 and self.dataclose[0] > self.sma[0]:
                self.order = self.buy()
        else:
            if self.dataclose[0] < self.sma[0]:
                self.order = self.sell()

# --- 2. BACKTESTING ENGINE ---
def run_backtest(strategy, ticker, from_date, to_date, cash=100000, commission=0.001):
    """
    Runs a backtest for a given strategy and stock.
    """
    cerebro = bt.Cerebro()
    cerebro.addstrategy(strategy)

    print(f"Downloading data for {ticker}...")
    df = yf.download(ticker, start=from_date, end=to_date, group_by='ticker')

    if isinstance(df.columns, pd.MultiIndex):
         df.columns = df.columns.droplevel(0)

    df.columns = [col.lower() for col in df.columns]

    data = bt.feeds.PandasData(dataname=df)
    cerebro.adddata(data)
    cerebro.broker.set_cash(cash)
    cerebro.broker.setcommission(commission=commission)
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe_ratio', riskfreerate=0.0)
    cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')

    print('Starting Portfolio Value: %.2f' % cerebro.broker.getvalue())
    results = cerebro.run()
    print('Final Portfolio Value: %.2f' % cerebro.broker.getvalue())

    final_results = results[0]
    sharpe = final_results.analyzers.sharpe_ratio.get_analysis()['sharperatio']
    returns = final_results.analyzers.returns.get_analysis()['rnorm100']
    drawdown = final_results.analyzers.drawdown.get_analysis()['max']['drawdown']

    print(f"\n--- PERFORMANCE METRICS for {ticker} ---")
    print(f"Sharpe Ratio: {sharpe:.3f}")
    print(f"Annualized Return: {returns:.2f}%")
    print(f"Maximum Drawdown: {drawdown:.2f}%")

    cerebro.plot(style='candlestick')

# --- 3. CONFIGURE AND RUN THE BACKTEST ---
if __name__ == '__main__':
    strategies = {
        '1': ('EMA Crossover', EMACrossoverStrategy),
        '2': ('SMA Crossover', SMACrossoverStrategy),
        '3': ('Bollinger Bands', BollingerBandStrategy),
        '4': ('RSI + MA Crossover', RsiMaStrategy),
        '5': ('Momentum + SMA Crossover', MomentumSMACrossoverStrategy),
    }

    print("Select a strategy to backtest:")
    for key, (name, _) in strategies.items():
        print(f"{key}: {name}")

    choice = input("Enter the number of the strategy: ")

    if choice in strategies:
        name, strategy_class = strategies[choice]
        print(f"\nRunning backtest for {name}...\n")

        stock_ticker = 'MSFT'
        start_date = datetime(2018, 1, 1)
        end_date = datetime(2024, 1, 1)
        initial_capital = 100000.0
        commission_fee = 0.001

        run_backtest(strategy_class, stock_ticker, start_date, end_date, initial_capital, commission_fee)
    else:
        print("Invalid choice. Please run the script again and select a valid number.")