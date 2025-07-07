# Portfolio Asset Allocator

This tool is an **interactive portfolio allocator** that helps users apply their investment strategy using simulations and visualizations.  
It is **not** an asset manager — you must define your own strategy, goals, and constraints beforehand. The allocator helps you *execute* that strategy more intelligently.

---

## What It Does

- Simulates thousands of random portfolio allocations
- Evaluates key metrics:
  - Expected annual return
  - Volatility (annualized standard deviation)
  - Sharpe & Sortino ratios
  - CAGR and Final Value
- Identifies the optimal allocation based on your chosen objective
- Visualizes:
  - Efficient frontier
  - Portfolio growth over time
  - Value contribution of each asset
  - Correlation heatmap of securities

---

## Structure (Sections 1–10)

1. **Libraries** – Import required packages  
2. **User Input & Data Fetching** – Select tickers, period, investment size  
3. **Data Cleaning** – Ensure all securities have valid OHLCV data  
4. **Preprocessing** – Scale/transform prices for returns  
5. **Simulation Engine** – Calculate performance metrics  
6. **Backtesting** – Evaluate how allocation performs on test data  
7. **Monte Carlo Simulation** – Run simulations to generate allocation candidates  
8. **Optimal Allocation** – Select best weights based on chosen metric  
9. **Comparison** – Quantify train vs test performance changes  
10. **Visualization** – Generate interactive charts (portfolio value, efficient frontier, contribution, correlation)

**Note:** When user input is required, it is clearly highlighted in the code using `# > USER INPUT`

---

## Setup

Install required packages:

```bash
pip install yfinance plotly pandas numpy matplotlib seaborn
