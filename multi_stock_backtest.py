import backtrader as bt
import yfinance as yf
import pandas as pd
from datetime import datetime
import numpy as np
from scipy.stats import linregress
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression

# --- 1. STRATEGY DEFINITIONS ---

class CrossSectionalMeanReversion(bt.Strategy):
    def __init__(self):
        self.stock_data = self.datas

    def prenext(self):
        self.next()

    def next(self):
        stock_returns = np.zeros(len(self.stock_data))

        for index, stock in enumerate(self.stock_data):
            stock_returns[index] = (stock.close[0] - stock.close[-1]) / stock.close[-1]

        market_return = np.mean(stock_returns)
        weights = -(stock_returns - market_return)
        weights = weights / np.sum(np.abs(weights))

        for index, stock in enumerate(self.stock_data):
            self.order_target_percent(stock, target=weights[index])

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

class MomentumPortfolioStrategy(bt.Strategy):
    params = (
        ('momentum_period', 90),
        ('rebalance_days', 5),
        ('num_top_stocks', 0.2),
    )

    def __init__(self):
        self.indicators = {}
        self.sorted_stocks = []
        self.counter = 0

        for stock in self.datas:
            self.indicators[stock] = {}
            self.indicators[stock]['momentum'] = Momentum(stock.close, period=self.p.momentum_period)

    def next(self):
        if self.counter % self.p.rebalance_days == 0:
            self.rebalance_portfolio()
        self.counter += 1

    def rebalance_portfolio(self):
        self.sorted_stocks = sorted(self.datas, key=lambda stock: self.indicators[stock]['momentum'][0], reverse=True)
        num_stocks_to_trade = int(np.ceil(len(self.datas) * self.p.num_top_stocks))
        top_stocks = self.sorted_stocks[:num_stocks_to_trade]

        for stock in self.datas:
            if stock in top_stocks:
                self.order_target_percent(stock, target=1.0 / num_stocks_to_trade)
            else:
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
        self.hedge_ratio = 0

    def next(self):
        prices1 = pd.Series(self.stock1.close.get(size=self.p.lookback))
        prices2 = pd.Series(self.stock2.close.get(size=self.p.lookback))

        if len(prices1) < self.p.lookback or len(prices2) < self.p.lookback:
            return

        hedge_ratio = sm.OLS(prices1, sm.add_constant(prices2)).fit().params.iloc[1]
        spread = prices1 - hedge_ratio * prices2
        zscore = (spread.iloc[-1] - spread.mean()) / spread.std()

        if not self.position:
            if zscore > self.p.entry_threshold:
                size1 = (self.broker.getvalue() * self.p.trade_size_percent) / self.stock1.close[0]
                self.sell(data=self.stock1, size=size1)
                self.buy(data=self.stock2, size=size1 * hedge_ratio)
            elif zscore < -self.p.entry_threshold:
                size1 = (self.broker.getvalue() * self.p.trade_size_percent) / self.stock1.close[0]
                self.buy(data=self.stock1, size=size1)
                self.sell(data=self.stock2, size=size1 * hedge_ratio)
        else:
            if abs(zscore) < self.p.exit_threshold:
                self.close(self.stock1)
                self.close(self.stock2)

class RollingLogisticRegressionStrategy(bt.Strategy):
    params = (
        ('lookback', 252),
        ('rebalance_days', 21),
    )

    def __init__(self):
        self.models = {}
        self.counter = 0

    def next(self):
        if self.counter % self.p.rebalance_days == 0:
            self.train_models()
        self.execute_trades()
        self.counter += 1

    def train_models(self):
        for stock in self.datas:
            df = pd.DataFrame({
                'returns': np.log(stock.close.get(size=self.p.lookback+1) / np.roll(stock.close.get(size=self.p.lookback+1), 1))[1:],
            })
            df['direction'] = np.where(df['returns'] > 0, 1, 0)
            df['lag1'] = df['returns'].shift(1)
            df.dropna(inplace=True)

            if len(df) < 20:
                continue

            X = df[['lag1']]
            y = df['direction']
            model = LogisticRegression()
            model.fit(X, y)
            self.models[stock] = model

    def execute_trades(self):
        for stock in self.datas:
            if stock in self.models:
                try:
                    feature = pd.DataFrame([np.log(stock.close[0] / stock.close[-1])], columns=['lag1'])
                    prediction = self.models[stock].predict(feature)[0]
                    if prediction == 1 and not self.getposition(stock):
                        self.order_target_percent(stock, target=1.0 / len(self.datas))
                    elif prediction == 0 and self.getposition(stock):
                        self.order_target_percent(stock, target=0.0)
                except:
                    pass

# --- 2. BACKTESTING ENGINE ---
def run_multi_stock_backtest(strategy, tickers, from_date, to_date, cash=100000, commission=0.001):
    cerebro = bt.Cerebro()
    cerebro.addstrategy(strategy)

    print(f"Downloading data for {', '.join(tickers)}...")
    for ticker in tickers:
        df = yf.download(ticker, start=from_date, end=to_date, group_by='ticker')
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(0)
        df.columns = [col.lower() for col in df.columns]
        data = bt.feeds.PandasData(dataname=df)
        cerebro.adddata(data, name=ticker)

    cerebro.broker.set_cash(cash)
    cerebro.broker.setcommission(commission=commission)
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe_ratio', riskfreerate=0.0)
    cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')

    print('Starting Portfolio Value: %.2f' % cerebro.broker.getvalue())
    results = cerebro.run()
    print('Final Portfolio Value: %.2f' % cerebro.broker.getvalue())

    final_results = results[0]
    sharpe = final_results.analyzers.sharpe_ratio.get_analysis().get('sharperatio')
    returns = final_results.analyzers.returns.get_analysis().get('rnorm100')
    drawdown = final_results.analyzers.drawdown.get_analysis().get('max', {}).get('drawdown')

    print(f"\n--- PERFORMANCE METRICS ---")
    print(f"Sharpe Ratio: {sharpe:.3f}" if sharpe is not None else "Sharpe Ratio: N/A")
    print(f"Annualized Return: {returns:.2f}%" if returns is not None else "Annualized Return: N/A")
    print(f"Maximum Drawdown: {drawdown:.2f}%" if drawdown is not None else "Maximum Drawdown: N/A")

    cerebro.plot(style='candlestick')

# --- 3. CONFIGURE AND RUN THE BACKTEST ---
if __name__ == '__main__':
    strategies = {
        '1': ('Cross-Sectional Mean Reversion', CrossSectionalMeanReversion, ['AAPL', 'MSFT', 'GOOG', 'AMZN']),
        '2': ('Momentum Portfolio', MomentumPortfolioStrategy, ['AAPL', 'MSFT', 'GOOG', 'AMZN']),
        '3': ('Pairs Trading (XOM/CVX)', PairsTradingStrategy, ['XOM', 'CVX']),
        '4': ('Rolling Logistic Regression', RollingLogisticRegressionStrategy, ['AAPL', 'MSFT', 'GOOG', 'AMZN']),
    }

    print("Select a strategy to backtest:")
    for key, (name, _, tickers) in strategies.items():
        print(f"{key}: {name} (Tickers: {', '.join(tickers)})")

    choice = input("Enter the number of the strategy: ")

    if choice in strategies:
        name, strategy_class, tickers = strategies[choice]
        print(f"\nRunning backtest for {name}...\n")

        start_date = datetime(2018, 1, 1)
        end_date = datetime(2024, 1, 1)
        initial_capital = 100000.0
        commission_fee = 0.001

        run_multi_stock_backtest(strategy_class, tickers, start_date, end_date, initial_capital, commission_fee)
    else:
        print("Invalid choice. Please run the script again and select a valid number.")