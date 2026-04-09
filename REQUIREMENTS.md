# Requirements Specification

## 1. Users and Environment

### 1.1 User Groups

| User Type          | Description | Key Permissions |
|--------------------|-------------|-----------------|
| Data Curator       |             |                 |
| Feature Analyst    |             |                 |
| Strategist         |             |                 |
| Backtester         |             |                 |
| Deployment Manager |             |                 |

### 1.2 Operating Environment

| Platform | Supported Environment |
|----------|-----------------------|
| macOS    | CLI                   |

## 2. Functional Requirements

### 2.1 Data Preprocessing

- Fundamental Data: Assets, Liabilities, Sales, Costs/earnings, Macro variables, …
- Market Data: Price/yield/implied volatility, Volume, Dividend/coupons, Open interest, Quotes/cancellations, Aggressor
  side, …
- Analytics: Analyst recommendations, Credit ratings, Earnings expectations, News sentiment, …
- Alternative Data: Satellite/CCTV images, Google searches, Twitter/chats, Metadata, … 

### 2.2 Feature Analysis

- Implementation of Core Factors

### 2.3 Strategy Research

- Portfolio strategies (value investing, long-short, trend following, mean reversion)
- Arbitrage strategies (market making, statistical arbitrage, event-driven arbitrage)
- Information-based trading (Buffett-style investing)

### 2.4 Backtesting

- Generate synthetic data
- Evaluate backtesting metrics in Jupyter notebooks
- Study reinforcement learning for HFT, where it may be effective

### 2.5 Automated Trading

- Enable online machine learning in live trading by supporting real-time data processing
- Connect to automated trading connectors for ultra-low-latency execution

### 2.6 Other Requirements

- User authentication
- Exception and failure handling
- 24/7 operation
- Log management
