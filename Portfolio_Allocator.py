#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# === 1. Libraries ===

# Required libraries: 
# pip install yfinance matplotlib plotly numpy pandas seaborn

import yfinance as yf             
import matplotlib.pyplot as plt   
import plotly.express as px       
import datetime as dt             
import numpy as np                
import pandas as pd               
import seaborn as sns             


# In[ ]:


# === 2. Constituent Securities ===

# > USER INPUT: Define the portfolio tickers (Yahoo Finance format)

tickers = ['URTH', 'XDWT.SW', '4GLD.DE', 'IXC']
print(f"Selected Tickers: {tickers}")


# In[ ]:


# === 3. Data Retrieval & Handling ===

# > USER INPUT: Define date range for historical data
start_date = dt.datetime(2013, 1, 1)
end_date = dt.datetime.now()
full_data_df = yf.download(tickers, start=start_date, end=end_date, auto_adjust=True)
full_data_df.columns = ['{}_{}'.format(col[0], col[1]) for col in full_data_df.columns]
# Fill missing values with its nearest value (financial data is very time dependent)
full_data_df = full_data_df.ffill().bfill()
full_data_df.head()


# In[ ]:


close_columns = [col for col in full_data_df.columns if col.startswith('Close_')]
close_df = full_data_df[close_columns].copy()
close_df.head()


# In[ ]:


# === 4. Simulation Preparation ===

# > USER INPUT: Define the cutoff date for splitting data into training and test sets

split_date = "2019-03-25"

def data_split(df):
    train_df = df.loc[:split_date].copy()
    test_df = df.loc[split_date:].copy()

    return train_df, test_df

close_train_df, close_test_df = data_split(close_df)


# In[ ]:


# --- Price Scaling ---
def price_scaling(df):
    scaled_prices_df = df.copy()
    for col in df.columns:
        #Scale prices so each security starts with 1
        scaled_prices_df[col] = df[col] / df[col].iloc[0] 
    return scaled_prices_df


# In[ ]:


# --- Weights --- 
import random

def generate_weights(n_securities):
    weights = []
    for i in range(n_securities):
        weights.append(random.random())
    weights = np.array(weights)
    # Normalize weights to ensure they all add up to 1
    weights /= np.sum(weights)
    return weights

n_securities = len(tickers)
weights = generate_weights(n_securities)


# In[ ]:


# --- Asset allocation ---

# > USER INPUT:
initial_investment = 100000

def asset_allocation(df, weights, initial_investment):
    scaled_df = price_scaling(df)

    # Compute total portfolio value per day by adding the daily position value together
    allocation = scaled_df * weights * initial_investment
    portfolio_value = allocation.sum(axis=1)
    # portfolio_value = scaled_df @ weights * initial_investment

    portfolio_df = allocation.copy()
    portfolio_df['Portfolio_Value'] = portfolio_value
    portfolio_df['Portfolio_Daily_Return'] = portfolio_value.pct_change()
    portfolio_df['Portfolio_Daily_Return'] = portfolio_df['Portfolio_Daily_Return'].fillna(0)

    return portfolio_df



# In[ ]:


# === 5. Simulation Engine ===

# --- Portfolio Metrics Evaluation ---

def portfolio_metrics(df, weights, initial_investment):
    portfolio_df = asset_allocation(df, weights, initial_investment)

    initial_value = portfolio_df['Portfolio_Value'].iloc[0]
    final_value = portfolio_df['Portfolio_Value'].iloc[-1]

    horizon = len(df)   
    days = 252          

    # Formula: CAGR = (Final / Initial)^(252 / N_days) - 1
    # "How much did the portfolio acutally grow per year over the time"
    CAGR = (final_value/initial_value) ** (days/horizon) - 1

    portfolio_returns_df = portfolio_df.drop(columns=['Portfolio_Value', 'Portfolio_Daily_Return'])
    portfolio_returns_df = portfolio_returns_df.pct_change()

    # Formula: E[R_p] = Σ(wᵢ * μᵢ) * 252
    # "Assuming historical daily returns continue, what annual return can be expected"
    expected_return = np.sum(weights * portfolio_returns_df.mean()) * days

    cov_matrix = portfolio_returns_df.cov() * days
    volatility = np.sqrt(weights.T @ cov_matrix @ weights)
    # volatility = (portfolio_returns_df @ weights).std() * np.sqrt(days)

    #Sortino ratio
    # Formula:
    #   Sortino = (E[R_p] - R_f) / σ_d
    #   where σ_d = downside deviation below target return

    # > USER INPUT: Set your personal target return (Minimum Acceptable Return) - it will be converted to daily as portfolio_returns are also daily
    target = 0.05 / days
    rf = 0.0257     # Annual risk-free rate → https://www.marketwatch.com/tools/markets/bonds

    # Matrix multiplication of daily returns with weight vector to get weighted daily return vector
    portfolio_daily_returns = portfolio_returns_df @ weights

    # Filter only returns below the target
    downside_returns = portfolio_daily_returns[portfolio_daily_returns < target]
    squared_diff = (downside_returns - target)** 2
    tdd = np.sqrt(squared_diff.mean()) * np.sqrt(days)

    sortino_ratio = (expected_return - rf) / tdd

    #Sharpe Ratio
    # Formula: Sharpe = (E[R_p] - R_f) / σ_p

    sharpe_ratio = (expected_return - rf) / volatility

    return expected_return, volatility, sortino_ratio, sharpe_ratio, final_value, CAGR * 100


# In[ ]:


# === 6. Monte Carlo Simulation ===

train_engine = lambda weights: portfolio_metrics(close_train_df, weights, initial_investment)

# > USER INPUT: Set the number of Simulation runs
sim_runs = 10000

#Placeholders for storing simulation results
weights_runs = np.zeros((sim_runs, n_securities))
final_value_runs = np.zeros(sim_runs)             
CAGR_runs = np.zeros(sim_runs)                    
expected_return_runs = np.zeros(sim_runs)         
volatility_runs = np.zeros(sim_runs)              
sortino_ratio_runs = np.zeros(sim_runs)           
sharpe_ratio_runs = np.zeros(sim_runs)            

# --- Monte Carlo loop ---
for i in range(sim_runs):
    # Step 1: Generate a new random set of porfolio weights
    weights = generate_weights(n_securities)
    # Step 2: Store the weights - select row i and all (:) columns
    weights_runs[i,:] = weights

    # Step 3: Evaluate perfolio performance on the training data
    expected_return_runs[i], volatility_runs[i], sortino_ratio_runs[i], sharpe_ratio_runs[i], final_value_runs[i], CAGR_runs[i] = train_engine(weights)

    # Step 4 (Optional): Live feedback
    print(f"Simulation Run = {i}")
    print(f"Weights ={weights_runs[i].round(4)}, "
      f"Final Value ={final_value_runs[i]:.2f}€, "
      f"Sharpe Ratio ={sharpe_ratio_runs[i]:.4f}, "
      f"Sortino Ratio ={sortino_ratio_runs[i]:.4f}, "
      f"Volatility ={volatility_runs[i] * 100:.2f}%, "
      f"CAGR ={CAGR_runs[i]:.2f}%, "
      f"Expected Annual Return ={expected_return_runs[i]:.2%}")



# In[ ]:


# === 7. Optimal Portfolio Selection ===

# NOTE: This is the core output of the Portfolio Asset Allocator — the optimal weights.

# The subsequent cells display the performance metrics and visualizations
# to help assess whether this portfolio (constructed using optimal weights) aligns with your personal investment goals.

# > USER INPUT: Select optimization criterion (e.g. maximize Sortino ratio, Sharpe ratio, minimize volatility, ...)
optimal_weights = weights_runs[sortino_ratio_runs.argmax(), :]

# Dictionary mapping constituent securities to their optimal weights
tickers_order = [col.replace("Close_", "") for col in close_df.columns if col.startswith("Close_")]
weights_dict = dict(zip(tickers_order, optimal_weights))

print("Optimal Portfolio Weights: ")
print(weights_dict)


# In[ ]:


#Re-evaluate the portfolio using the optimal weights
train_metrics = train_engine(optimal_weights)
train_return, train_volatility, train_sharpe, train_sortino, train_value, train_CAGR = train_metrics

print(f"\nTraining Metrics corresponding to the selected optimization criterion")
print(f"- Expected Annual Portfolio Return:         {train_return * 100:.2f}%")
print(f"- Compounded Annual Growth Rate (CAGR):     {train_CAGR:.2f}%")
print(f"- Final Portfolio Value:                    {train_value:,.2f}€")
print(f"- Sortino Ratio:                            {train_sortino:.4f}")
print(f"- Sharpe Ratio:                             {train_sharpe:.4f}")
print(f"- Portfolio Volatility (Annualized):        {train_volatility * 100:.2f}%")


# In[ ]:


# === 8. Backtesting ===

test_engine = lambda weights: portfolio_metrics(close_test_df, weights, initial_investment)

# Compute portfolio metrics using optimal weights on unseen data (test data)
test_metrics = test_engine(optimal_weights)
test_return, test_volatility, test_sortino, test_sharpe, test_value, test_CAGR = test_metrics

print(f"\nTesting Metrics corresponding to the selected optimization criterion")
print(f"- Expected Annual Portfolio Return:         {test_return * 100:.2f}%")
print(f"- Compounded Annual Growth Rate (CAGR):     {test_CAGR:.2f}%")
print(f"- Final Portfolio Value:                    {test_value:.2f}€")
print(f"- Sortino Ratio:                            {test_sortino:.2f}")
print(f"- Sharpe Ratio :                            {test_sharpe:.2f}")
print(f"- Portfolio Volatility (Annualized):        {test_volatility * 100:.2f}%")


# In[ ]:


# ==== 9. Comparison ===

# Compare the training and testing metrics to analyze portfolio performance changes. 
# USEAGE: Trend Analysis
# INSIGHT: Refinement of strategy
def comparison(train_metrics, test_metrics):

    # Define metric metadata in one place
    metrics_info = [
        {'name': 'Expected Annual Return', 'type': 'percent'},
        {'name': 'Volatility',              'type': 'percent'},
        {'name': 'Sortino Ratio',           'type': 'ratio'},
        {'name': 'Sharpe Ratio',            'type': 'ratio'},
        {'name': 'Final Portfolio Value',   'type': 'currency'},
        {'name': 'CAGR',                    'type': 'pp'},  # pp = percentage points
    ]

    print("\n--- Metric Evolution from Train → Test ---")
    print("-" * 60)

    for info, train, test in zip(metrics_info, train_metrics, test_metrics):
        diff = test - train
        direction = "increased" if diff > 0 else "decreased" if diff < 0 else "remained unchanged"
        abs_diff = abs(diff)

        # Format based on type
        if info['type'] == 'percent':
            change_str = f"{abs_diff * 100:.2f}%"
        elif info['type'] == 'pp':
            change_str = f"{abs_diff:.2f} percentage points"
        elif info['type'] == 'currency':
            change_str = f"{abs_diff:,.2f}€"
        else:  # 'ratio'
            change_str = f"{abs_diff:.4f}"

        print(f"{info['name']}: {direction} by {change_str}.")

comparison = comparison(train_metrics, test_metrics)
comparison


# In[ ]:


# === 10. Visualization ===

# --- Plot Monte Carlo Simulation as Efficient Frontier ---

# DataFrame with simulation results needed for Efficient Frontier
efficient_df = pd.DataFrame({'Volatility': volatility_runs.tolist(), 
                             'Return': expected_return_runs.tolist(), 
                             'Sortino Ratio': sortino_ratio_runs.tolist(),
                             'Value': final_value_runs.tolist()
})

import plotly.io as pio
import plotly.graph_objects as go 

# Plot Efficient Frontier 
fig = px.scatter(efficient_df, 
                 x='Volatility', 
                 y='Return', 
                 color='Sortino Ratio',
                 hover_data=['Sortino Ratio'], 
                 size='Value',
                 title='Markowitz Efficient Frontier',
)

#Highlight the optimal portfolio point (based on selected criterion)
fig.add_trace(go.Scatter(
    x=[train_volatility],
    y=[train_return],
    mode='markers',
    name='Optimal Point',
    marker=dict(size=40, color='red'),
    customdata=[[train_sortino, train_value]],
    hovertemplate=(
        'Volatility: %{x:.2%}<br>' +
        'Return: %{y:.2%}<br>' +
        'Sortino Ratio: %{customdata[0]:.2f}<br>' +
        'Value: %{customdata[1]:,.2f}€<br>' +
        '<extra>Optimal Point</extra>'
    )
))

fig.update_layout(coloraxis_colorbar = dict(y = 0.7, dtick = 5))
fig.update_layout({'plot_bgcolor': "white"})
fig.show()


# In[ ]:


# --- Plot Portfolio Performance as interactive chart ---

# >USER INPUT: Choose the time period to visualize by changing the DataFrame below:
# - `close_train_df` → to visualize only the training period
# - `close_test_df`  → to visualize only the testing period
# - `close_df`       → to visualize the full period 
plot_portfolio_df = asset_allocation(close_df, optimal_weights, initial_investment)

#Ensure DataFrame index is datetime for correct plotting
plot_portfolio_df.index = pd.to_datetime(plot_portfolio_df.index)

volatility = plot_portfolio_df['Portfolio_Daily_Return'].std() * np.sqrt(252)

# Create interactive line chart
fig = go.Figure()

fig.add_trace(go.Scatter(
    x=plot_portfolio_df.index,
    y=plot_portfolio_df['Portfolio_Value'],
    mode='lines',
    name='Portfolio Value',
    line=dict(color='royalblue', width=2),
    hovertemplate='Date: %{x}<br>Value: €%{y:,.2f}<extra></extra>'
))

fig.update_layout(
    title=(
        f"Interactive Portfolio Value Over Time<br>"
        f"<sub>Initial: €{plot_portfolio_df['Portfolio_Value'].iloc[0]:,.2f} | "
        f"Final: €{plot_portfolio_df['Portfolio_Value'].iloc[-1]:,.2f} | "
        f"Annual Volatility: {volatility:.2%}</sub>"
    ),
    xaxis_title="Date",
    yaxis_title="Portfolio Value (€)",
    template='plotly_white',
    hovermode='x unified'
)

fig.show()



# In[ ]:


# Calculate daily returns for each security

# Remove prefix 'Close_', so only the ticker symbols remain
plot_portfolio_df.rename(columns=lambda x: x.replace("Close_", ""), inplace=True)
# Exclude 'Portfolio_Value' and 'Portfolio_Daily_Return' from the return calculation
security_returns = plot_portfolio_df.iloc[:, :-2].pct_change().fillna(0)

# --- Interactive Stacked Area Chart for Value Contributions ---

# Value Contribution == "What drives your portfolio"
# USEAGE: Helps spot dominant assets
# INSIGHT: Reveals imbalances → apply targeted countermeasurements 
fig_value = go.Figure()

# Loop through each individual security's allocation over time
for col in plot_portfolio_df.columns[:-2]:
    fig_value.add_trace(go.Scatter(
        x=plot_portfolio_df.index,
        y=plot_portfolio_df[col],
        mode='lines',
        stackgroup='one',     
        name=col              
    ))

fig_value.update_layout(
    title='Portfolio Value Contribution Over Time',
    xaxis_title='Date',
    yaxis_title='Value (€)',
    legend_title='Securities',
    hovermode='x unified'
)

# --- Correlation Heatmap of Securities ---

correlation_matrix = security_returns.corr()
sns.set(style="white")

# Create a matplotlib figure
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap="magma", fmt=".2f", square=True, cbar_kws={'label': 'Correlation'})

plt.title("Correlation Heatmap of Portfolio Constituents", fontsize=14)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()

fig_value.show()
plt.show()


# In[ ]:




