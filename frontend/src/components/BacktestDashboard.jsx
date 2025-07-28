import React, { useState, useEffect, useMemo } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, BarChart, Bar, ReferenceDot } from 'recharts';
import { TrendingUp, TrendingDown, DollarSign, BarChart3, Settings, Play, AlertCircle, Calendar, Tag, ArrowUpCircle, ArrowDownCircle, XCircle, Percent } from 'lucide-react';

// --- VALUATION CHART COMPONENT ---
const ValuationChart = ({ valuationData, signals, tickers }) => {
  if (!valuationData || valuationData.length === 0) {
    return <p className="text-slate-400">No valuation data available to display.</p>;
  }

  const lineKeys = Object.keys(valuationData[valuationData.length - 1]).filter(key => key !== 'date' && valuationData[valuationData.length - 1][key] !== null);
  const priceKeys = lineKeys.filter(key => key.startsWith('price_'));
  const indicatorKeys = lineKeys.filter(key => !key.startsWith('price_'));

  const colors = useMemo(() => {
    const colorPalette = ["#3b82f6", "#82ca9d", "#ffc658", "#ff8042", "#00C49F", "#FFBB28", "#e06262", "#d879e0"];
    const generatedColors = {};
    priceKeys.forEach((key, index) => {
        generatedColors[key] = colorPalette[index % colorPalette.length];
    });
    indicatorKeys.forEach((key, index) => {
      generatedColors[key] = colorPalette[(priceKeys.length + index) % colorPalette.length];
    });
    return generatedColors;
  }, [indicatorKeys, priceKeys]);

  const CustomTooltip = ({ active, payload, label }) => {
    if (active && payload && payload.length) {
      return (
        <div className="bg-slate-700/80 backdrop-blur-sm p-4 border border-slate-600 rounded-lg shadow-lg">
          <p className="label text-slate-300">{`${new Date(label).toLocaleDateString()}`}</p>
          {payload.map(pld => (
            <p key={pld.name} style={{ color: pld.color }}>
              {`${pld.name}: ${typeof pld.value === 'number' ? pld.value.toFixed(2) : 'N/A'}`}
            </p>
          ))}
        </div>
      );
    }
    return null;
  };

  const SignalDot = ({ cx, cy, payload, signal }) => {
    if (!signal) return null;
    const iconProps = { x: cx - 10, y: cy - 10, width: 20, height: 20, strokeWidth: 1.5 };
    if (signal.type === 'BUY') return <ArrowUpCircle {...iconProps} className="text-green-400" fill="#1F2937" />;
    if (signal.type === 'SELL') return <ArrowDownCircle {...iconProps} className="text-red-400" fill="#1F2937" />;
    if (signal.type === 'CLOSE') return <XCircle {...iconProps} className="text-yellow-400" fill="#1F2937" />;
    return null;
  };

  return (
    <ResponsiveContainer width="100%" height={400}>
      <LineChart data={valuationData} margin={{ top: 5, right: 20, left: 20, bottom: 5 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
        <XAxis dataKey="date" stroke="#9CA3AF" tick={{ fontSize: 12 }} tickFormatter={(value) => new Date(value).toLocaleDateString()} />
        <YAxis yAxisId="left" stroke="#9CA3AF" tick={{ fontSize: 12 }} tickFormatter={(value) => value.toFixed(2)} domain={['auto', 'auto']} />
        <Tooltip content={<CustomTooltip />} />
        <Legend />

        {priceKeys.map(key => (
          <Line key={key} yAxisId="left" type="monotone" dataKey={key} name={key.replace('price_', '')} stroke={colors[key]} strokeWidth={2} dot={false} activeDot={{ r: 6 }} connectNulls={true} />
        ))}

        {indicatorKeys.map(key => (
          <Line key={key} yAxisId="left" type="monotone" dataKey={key} name={key} stroke={colors[key]} strokeWidth={1.5} strokeDasharray="5 5" dot={false} connectNulls={true} />
        ))}

        {signals.map((signal, index) => (
           <ReferenceDot key={index} yAxisId="left" x={signal.date} y={signal.price} r={10} ifOverflow="extendDomain" shape={<SignalDot signal={signal} />} />
        ))}
      </LineChart>
    </ResponsiveContainer>
  );
};


const BacktestDashboard = () => {
  const [strategies, setStrategies] = useState({});
  const [selectedStrategy, setSelectedStrategy] = useState('1');
  const [tickers, setTickers] = useState('MSFT');
  const [startDate, setStartDate] = useState('2022-01-01');
  const [endDate, setEndDate] = useState('2024-01-01');
  const [isRunning, setIsRunning] = useState(false);
  const [results, setResults] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    const fetchStrategies = async () => {
      try {
        const response = await fetch('http://localhost:5000/api/strategies');
        const data = await response.json();
        setStrategies(data);
        if (data['1']) setTickers(data['1'].tickers.join(','));
      } catch (err) {
        setError('Failed to fetch strategies from the backend. Is the Flask server running?');
      }
    };
    fetchStrategies();
  }, []);

  useEffect(() => {
    if (strategies[selectedStrategy]) {
      setTickers(strategies[selectedStrategy].tickers.join(','));
    }
  }, [selectedStrategy, strategies]);

  const runBacktest = async () => {
    setIsRunning(true);
    setError(null);
    setResults(null);
    
    try {
      const start = new Date(startDate);
      const end = new Date(endDate);
      if (start >= end) {
        setError('Start date must be before end date');
        setIsRunning(false);
        return;
      }
      
      const tickerList = tickers.split(',').map(t => t.trim().toUpperCase()).filter(t => t);
      if (tickerList.length === 0) {
        setError('Please enter at least one valid ticker symbol.');
        setIsRunning(false);
        return;
      }
      
      const response = await fetch('http://localhost:5000/api/backtest', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          strategy_id: selectedStrategy,
          tickers: tickerList,
          start_date: startDate,
          end_date: endDate
        })
      });
      
      const result = await response.json();
      
      if (result.status === 'success') {
        setResults(result);
        setError(null);
      } else {
        setError(result.error || 'An unknown error occurred during the backtest.');
        setResults(null);
      }
    } catch (error) {
      console.error('API call failed:', error);
      setError('Failed to connect to the backend. Please ensure the Flask server is running on port 5000.');
      setResults(null);
    }
    
    setIsRunning(false);
  };

  const formatCurrency = (value) => new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD' }).format(value);
  const formatPercentage = (value) => `${value > 0 ? '+' : ''}${value.toFixed(2)}%`;

  const getStrategyDescription = (strategyId) => {
    const descriptions = {
      '1': { title: 'EMA Crossover', description: 'Buys when a short-term Exponential Moving Average (EMA) crosses above a long-term EMA.' },
      '2': { title: 'SMA Crossover', description: 'Buys when a short-term Simple Moving Average (SMA) crosses above a long-term SMA.' },
      '3': { title: 'Bollinger Bands', description: 'A mean-reversion strategy that buys when the price hits the lower band and sells when it crosses the middle band.' },
      '4': { title: 'RSI + MA Crossover', description: 'Combines moving average trend-following with an RSI filter to buy in uptrends when the asset is oversold.' },
      '5': { title: 'Momentum + SMA', description: 'Buys when momentum is positive and the price is above a long-term SMA, confirming the trend.' },
      '6': { title: 'Cross-Sectional Mean Reversion', description: 'A portfolio strategy that shorts recent winners and buys recent losers across a universe of stocks.' },
      '7': { title: 'Momentum Portfolio', description: 'A portfolio strategy that buys the top-performing stocks based on recent momentum.' },
      '8': { title: 'Pairs Trading', description: 'A market-neutral strategy that trades the spread between two historically correlated stocks.' },
      '9': { title: 'Rolling Logistic Regression', description: 'A machine learning approach that uses past returns to predict the future direction of a stock.' }
    };
    return descriptions[strategyId] || { title: 'Unknown Strategy', description: 'N/A' };
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-blue-900 to-slate-900 text-white font-sans">
      <div className="container mx-auto px-4 py-8">
        <header className="text-center mb-10">
          <h1 className="text-4xl md:text-5xl font-bold mb-3 bg-gradient-to-r from-blue-400 to-purple-400 bg-clip-text text-transparent">
            Algorithmic Trading Backtester
          </h1>
          <p className="text-lg text-slate-300">Analyze trading strategies with historical market data.</p>
        </header>

        <main>
          <div className="bg-slate-800/50 backdrop-blur-sm rounded-2xl p-6 border border-slate-700 mb-8">
            <h2 className="text-2xl font-bold mb-4 flex items-center gap-2"><Settings />Configuration</h2>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
              <div>
                <label className="block text-sm font-medium text-slate-300 mb-2 h-6 flex items-center">Strategy</label>
                <select value={selectedStrategy} onChange={(e) => setSelectedStrategy(e.target.value)} className="w-full p-3 bg-slate-700 border border-slate-600 rounded-lg focus:ring-2 focus:ring-blue-500">
                  {Object.entries(strategies).map(([key, s]) => <option key={key} value={key}>{s.name}</option>)}
                </select>
              </div>
              <div>
                <label className="block text-sm font-medium text-slate-300 mb-2 h-6 flex items-center gap-2">
                    <Tag className="w-4 h-4" />
                    Tickers
                </label>
                <input type="text" value={tickers} onChange={(e) => setTickers(e.target.value.toUpperCase())} placeholder="AAPL,MSFT,GOOGL" className="w-full p-3 bg-slate-700 border border-slate-600 rounded-lg focus:ring-2 focus:ring-blue-500" />
              </div>
              <div>
                <label className="block text-sm font-medium text-slate-300 mb-2 h-6 flex items-center">Start Date</label>
                <input type="date" value={startDate} onChange={(e) => setStartDate(e.target.value)} className="w-full p-3 bg-slate-700 border border-slate-600 rounded-lg focus:ring-2 focus:ring-blue-500" />
              </div>
              <div>
                <label className="block text-sm font-medium text-slate-300 mb-2 h-6 flex items-center">End Date</label>
                <input type="date" value={endDate} onChange={(e) => setEndDate(e.target.value)} className="w-full p-3 bg-slate-700 border border-slate-600 rounded-lg focus:ring-2 focus:ring-blue-500" />
              </div>
            </div>
            <div className="mt-4 text-sm text-slate-400 bg-slate-700/50 p-3 rounded-lg">
                <p className="font-semibold text-blue-400">{getStrategyDescription(selectedStrategy).title}</p>
                <p>{getStrategyDescription(selectedStrategy).description}</p>
            </div>
          </div>

          <div className="text-center mb-8">
            <button onClick={runBacktest} disabled={isRunning} className="bg-gradient-to-r from-blue-500 to-purple-600 hover:from-blue-600 hover:to-purple-700 disabled:opacity-50 disabled:cursor-not-allowed px-8 py-4 rounded-xl font-bold text-lg transition-all shadow-lg hover:shadow-xl flex items-center gap-2 mx-auto">
              {isRunning ? <><div className="animate-spin rounded-full h-5 w-5 border-b-2 border-white"></div>Running...</> : <><Play />Run Backtest</>}
            </button>
          </div>

          {error && (
            <div className="mb-8 bg-red-900/50 backdrop-blur-sm rounded-2xl p-4 border border-red-700 flex items-center gap-3">
              <AlertCircle className="w-6 h-6 text-red-400" />
              <div><h3 className="font-bold text-red-400">Error</h3><p className="text-red-300">{error}</p></div>
            </div>
          )}

          {results && (
            <div className="space-y-8">
              <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-5 gap-6">
                  {[
                      {label: "Total Return", value: formatPercentage(results.metrics.totalReturn), Icon: results.metrics.totalReturn >= 0 ? TrendingUp : TrendingDown, color: results.metrics.totalReturn >= 0 ? 'text-green-400' : 'text-red-400'},
                      {label: "Sharpe Ratio", value: results.metrics.sharpeRatio.toFixed(2), Icon: BarChart3, color: 'text-blue-400'},
                      {label: "Annualised Return", value: formatPercentage(results.metrics.annualizedReturn), Icon: Percent, color: results.metrics.annualizedReturn >= 0 ? 'text-green-400' : 'text-red-400'},
                      {label: "Max Drawdown", value: `${results.metrics.maxDrawdown.toFixed(2)}%`, Icon: TrendingDown, color: 'text-red-400'},
                      {label: "Final Value", value: formatCurrency(results.metrics.finalValue), color: 'text-white'}
                  ].map(metric => (
                      <div key={metric.label} className="bg-slate-800/50 backdrop-blur-sm rounded-2xl p-6 border border-slate-700 flex items-center justify-between">
                          <div>
                              <p className="text-sm text-slate-400">{metric.label}</p>
                              <p className={`text-2xl font-bold ${metric.color}`}>{metric.value}</p>
                          </div>
                          {metric.Icon && <metric.Icon className={`w-8 h-8 ${metric.color}`} />}
                      </div>
                  ))}
              </div>
              
              <div className="bg-slate-800/50 backdrop-blur-sm rounded-2xl p-6 border border-slate-700">
                <h3 className="text-2xl font-bold mb-6">Trade Signals & Indicators</h3>
                <ValuationChart 
                  valuationData={results.valuationData} 
                  signals={results.signals} 
                  tickers={results.tickers}
                />
              </div>

              <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
                <div className="bg-slate-800/50 backdrop-blur-sm rounded-2xl p-6 border border-slate-700">
                  <h3 className="text-2xl font-bold mb-6">Portfolio Performance</h3>
                  <ResponsiveContainer width="100%" height={300}>
                    <LineChart data={results.data}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                      <XAxis dataKey="date" stroke="#9CA3AF" tick={{ fontSize: 12 }} tickFormatter={(val) => new Date(val).toLocaleDateString()} />
                      <YAxis stroke="#9CA3AF" tick={{ fontSize: 12 }} tickFormatter={(val) => formatCurrency(val)} yAxisId="left" orientation="left" />
                      <Tooltip contentStyle={{ backgroundColor: '#1F2937', border: '1px solid #374151', borderRadius: '8px' }} formatter={(val) => [formatCurrency(val), 'Portfolio Value']} />
                      <Line yAxisId="left" type="monotone" dataKey="portfolioValue" stroke="#3B82F6" strokeWidth={2} dot={false} />
                    </LineChart>
                  </ResponsiveContainer>
                </div>

                <div className="bg-slate-800/50 backdrop-blur-sm rounded-2xl p-6 border border-slate-700">
                  <h3 className="text-2xl font-bold mb-6">Drawdown</h3>
                  <ResponsiveContainer width="100%" height={300}>
                    <BarChart data={results.data}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                      <XAxis dataKey="date" stroke="#9CA3AF" tick={{ fontSize: 12 }} tickFormatter={(val) => new Date(val).toLocaleDateString()} />
                      <YAxis stroke="#9CA3AF" tick={{ fontSize: 12 }} tickFormatter={(val) => `${val}%`} />
                      <Tooltip contentStyle={{ backgroundColor: '#1F2937', border: '1px solid #374151', borderRadius: '8px' }} formatter={(val) => [`-${val.toFixed(2)}%`, 'Drawdown']} />
                      <Bar dataKey="drawdown" name="Drawdown" fill="#EF4444" barSize={20} />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </div>

              {/* Restored Backtest Information Section */}
              <div className="bg-slate-800/50 backdrop-blur-sm rounded-2xl p-6 border border-slate-700">
                <h3 className="text-2xl font-bold mb-4">Backtest Summary</h3>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-slate-300">
                  <div>
                    <p><span className="font-semibold text-slate-400">Strategy:</span> {strategies[selectedStrategy]?.name}</p>
                    <p><span className="font-semibold text-slate-400">Tickers:</span> {results.tickers.join(', ')}</p>
                    <p><span className="font-semibold text-slate-400">Period:</span> {startDate} to {endDate}</p>
                  </div>
                  <div>
                    <p><span className="font-semibold text-slate-400">Initial Capital:</span> {formatCurrency(results.metrics.initialValue)}</p>
                    <p><span className="font-semibold text-slate-400">Commission:</span> 0.1%</p>
                    <p><span className="font-semibold text-slate-400">Final Portfolio Value:</span> {formatCurrency(results.metrics.finalValue)}</p>
                  </div>
                </div>
              </div>

            </div>
          )}
        </main>
      </div>
    </div>
  );
};

export default BacktestDashboard;
