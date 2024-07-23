import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm
from django.shortcuts import render
from django.http import JsonResponse
from .forms import HeatmapForm
from io import BytesIO
import base64
import matplotlib.gridspec as gridspec

# Set the Matplotlib backend to 'Agg'
plt.switch_backend('Agg')

#Black Scholes model for Call option pricing
def BS_CALL(S, K, T, r, sigma):
    d1 = (np.log(S/K) + (r + sigma**2/2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r*T)* norm.cdf(d2)


#Black Scholes model for Put option pricing
def BS_PUT(S, K, T, r, sigma):
    d1 = (np.log(S/K) + (r + sigma**2/2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return K * np.exp(-r*T) * norm.cdf(-d2) - S * norm.cdf(-d1)

def get_random_color(r, g, b):
    r = np.random.uniform(r[0], r[1])
    g = np.random.uniform(g[0], g[1])
    b = np.random.uniform(b[0], b[1])
    return (r, g, b)

def generate_Outputs(stock_price, strike_price, time, risk_free_interest_rate, volatility, n_simulations):
    row_labels = np.arange(volatility - .1, volatility + .11, .02)
    row_labels = list(np.around(np.array(row_labels), 2))
    column_labels = np.arange(.9 * strike_price, 1.1 * strike_price + .001, .2 * strike_price / 10)
    column_labels = list(np.around(np.array(column_labels), 0))
    Call_price = BS_CALL(stock_price, strike_price, time, risk_free_interest_rate, volatility)
    Put_price = BS_PUT(stock_price, strike_price, time, risk_free_interest_rate, volatility)
    
    #calculate trading days in a period
    start_date = pd.Timestamp.today()
    end_date = start_date + pd.DateOffset(days=time*365)
    date_range = pd.bdate_range(start=start_date, end=end_date)
    trading_days = len(date_range)
    
    dt = time / trading_days
    Simulations = np.zeros((trading_days + 1, n_simulations))
    call_option_simulation = np.zeros((trading_days + 1, n_simulations))
    put_option_simulation = np.zeros((trading_days + 1, n_simulations))
    Simulations[0] = stock_price
    call_option_simulation[0] = Call_price
    put_option_simulation[0] = Put_price
    Optimal_stop = int(trading_days*.8)
    
    #perform Monte Carlo Simulations using Brownian Motion Model
    call_threshold_line = stock_price + Call_price
    put_threshold_line = stock_price - Put_price
    call_count_above_threshold = np.zeros(n_simulations, dtype=int)
    put_count_below_threshold = np.zeros(n_simulations, dtype=int)
    max_call_prices = np.zeros(n_simulations)
    max_call_prices.fill(Call_price)
    max_call_prices_binary = np.zeros(n_simulations)
    max_put_prices = np.zeros(n_simulations)
    max_put_prices.fill(Put_price)
    max_put_prices_binary = np.zeros(n_simulations)
    
    for t in range(1, trading_days + 1):
        Z = np.random.standard_normal(n_simulations)
        Simulations[t] = Simulations[t-1] * np.exp((risk_free_interest_rate - 0.5 * volatility**2) * dt + volatility * np.sqrt(dt) * Z)
        call_count_above_threshold += (Simulations[t] > call_threshold_line).astype(int)
        put_count_below_threshold += (Simulations[t] < put_threshold_line).astype(int)
        call_option_simulation[t] = BS_CALL(Simulations[t], strike_price, (trading_days - t + 1)/365, risk_free_interest_rate, volatility)
        put_option_simulation[t] = BS_PUT(Simulations[t], strike_price, (trading_days - t + 1)/365, risk_free_interest_rate, volatility)
        if t < Optimal_stop:
            max_call_prices = np.where(call_option_simulation[t] > max_call_prices, call_option_simulation[t], max_call_prices )
            max_put_prices = np.where(put_option_simulation[t] > max_put_prices, put_option_simulation[t], max_put_prices)
        else:
            temp1 = max_call_prices
            temp2 = max_put_prices
            max_call_prices = np.where((call_option_simulation[t] > max_call_prices) & (max_call_prices_binary == 0), call_option_simulation[t], max_call_prices)
            max_call_prices_binary = np.where(temp1 != max_call_prices, 1, max_call_prices_binary)
            max_put_prices = np.where((put_option_simulation[t] > max_put_prices) & (max_put_prices_binary == 0), put_option_simulation[t], max_put_prices)
            max_put_prices_binary = np.where(temp2 != max_put_prices, 1, max_put_prices_binary)
        if t == trading_days:
            max_call_prices = np.where(max_call_prices_binary == 0, call_option_simulation[t], max_call_prices)
            max_put_prices = np.where(max_put_prices_binary == 0, put_option_simulation[t], max_put_prices)

    # Create pd df for Call Option Prices
    Call_heat_map = pd.DataFrame(index=row_labels, columns=column_labels)
    for V in row_labels:
        for SP in column_labels:
            Call_heat_map.loc[V, SP] = BS_CALL(stock_price, SP, time, risk_free_interest_rate, V)
    Call_heat_map = Call_heat_map.astype(float)

    # Create pd df for Put Option Prices
    Put_heat_map = pd.DataFrame(index=row_labels, columns=column_labels)
    for V in row_labels:
        for SP in column_labels:
            Put_heat_map.loc[V, SP] = BS_PUT(stock_price, SP, time, risk_free_interest_rate, V)
    Put_heat_map = Put_heat_map.astype(float)
    
    
    #create threshold lines to visualize profitable excersizing
    profit_thresholds_call = []
    profit_thresholds_put = []
    for t in range(trading_days):
        call_threshold = BS_CALL(Simulations[0, 0], strike_price, (time - (t * dt)), risk_free_interest_rate, volatility)
        put_threshold = BS_PUT(Simulations[0, 0], strike_price, (time - (t * dt)), risk_free_interest_rate, volatility)
        profit_thresholds_call.append(call_threshold+stock_price)
        profit_thresholds_put.append(-put_threshold+stock_price)

    # Create figure for both heatmaps and the simulation chart
    fig = plt.figure(figsize=(12, 18))
    gs = gridspec.GridSpec(3, 2, height_ratios=[1, 1, 1])
    
    # subplots: 2 heat maps on the first row followed by stock price simulations, then options pricing based off of those simulations
    call_plot = plt.subplot(gs[0, 0])
    put_plot = plt.subplot(gs[0, 1])
    MonteCarlo_plot = plt.subplot(gs[1, :])
    Options_plot = plt.subplot(gs[2, :])
    
    fig.patch.set_facecolor('#222222') 

    # Call Heatmap
    sns.heatmap(Call_heat_map, annot=True, fmt=".1f", cmap="YlGnBu", cbar=False, ax=call_plot)
    call_plot.set_title("Call Option Prices (Black-Scholes)", color='white')
    call_plot.set_xlabel("Strike Price", color='white')
    call_plot.set_ylabel("Volatility", color='white') 
    call_plot.tick_params(colors='white')

    # Put Heatmap
    sns.heatmap(Put_heat_map, annot=True, fmt=".1f", cmap="YlGnBu", cbar=False, ax=put_plot)
    put_plot.set_title("Put Option Prices (Black-Scholes)", color='white')
    put_plot.set_xlabel("Strike Price", color='white') 
    put_plot.set_ylabel("Volatility", color='white')
    put_plot.tick_params(colors='white')
    
    # Monte Carlo line Graph(s)
    for i in range(1, n_simulations):
        MonteCarlo_plot.plot(Simulations[:, i], color=get_random_color((0,0), (0, 1), (0.8, 1)) ,lw=.7)
    MonteCarlo_plot.plot(Simulations[:, 0], label = "Simulations", color="#00FFFF" ,lw=.7)
    # MonteCarlo_plot.plot(profit_thresholds_call, label='Call Profit Threshold', color='red', linestyle='--')
    # MonteCarlo_plot.plot(profit_thresholds_put, label='Put Profit Threshold', color='yellow', linestyle='--')
    # MonteCarlo_plot.axhline(y=call_threshold_line, label='Call Profit Threshold', color='red', linestyle='--')
    # MonteCarlo_plot.axhline(y=put_threshold_line, label='Put Profit Threshold', color='yellow', linestyle='--')
    MonteCarlo_plot.set_title('Monte Carlo Simulation of Stock Price Paths', color='white')
    MonteCarlo_plot.set_xlabel('Trading days', color='white')
    MonteCarlo_plot.set_ylabel('Stock Price', color='white')
    MonteCarlo_plot.tick_params(colors='white')
    MonteCarlo_plot.grid(True)
    MonteCarlo_plot.set_facecolor('#222222')
    MonteCarlo_plot.legend(loc='upper left', labelcolor="w", facecolor='grey')
    
    # Options line Graph(s)
    
    for i in range(1, n_simulations):
        Options_plot.plot(call_option_simulation[:, i], color="#00FFFF" ,lw=.7)
        Options_plot.plot(put_option_simulation[:, i], color="r" ,lw=.7 )
    Options_plot.plot(call_option_simulation[:, 0], label="Call Options", color="#00FFFF" ,lw=.7)
    Options_plot.plot(put_option_simulation[:, 0], label="Put Options",color="r" ,lw=.7)
    Options_plot.set_title('Options Pricing based on simulations', color='white')
    Options_plot.set_xlabel('Trading days', color='white')
    Options_plot.set_ylabel('Option Price', color='white')
    Options_plot.tick_params(colors='white')
    Options_plot.grid(True)
    Options_plot.set_facecolor('#222222')
    Options_plot.legend(loc='upper left', labelcolor="w", facecolor='grey')
    

    plt.tight_layout()

    # Save plot
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    image_png = buffer.getvalue()
    buffer.close()
    plt.clf()  # Clear
    
    final_prices = Simulations[-1]
    profitable_simulations = np.sum(final_prices >= stock_price) / n_simulations
    avg_returns = np.mean(final_prices / Simulations[0] - 1)
    
    metrics = {
        'profitable_simulations': round(100 * profitable_simulations, 2),
        'avg_returns': round(100 * avg_returns, 2),
        'count_above_threshold': round(100 * int(np.sum(call_option_simulation[-1] > Call_price))/n_simulations, 2),
        'put_below_threshold': round(100 * int(np.sum(put_option_simulation[-1] > Put_price))/n_simulations, 2),
        'call_expire_worthless': round(100 * int(np.sum(call_option_simulation[-1] <= 1))/n_simulations, 2),
        'put_expire_worthless': round(100 * int(np.sum(put_option_simulation[-1] <= 1))/n_simulations, 2),
        'call_average_returns': round(100*np.sum(call_option_simulation[-1] - Call_price)/n_simulations,2),
        'put_average_returns': round(100*np.sum(put_option_simulation[-1] - Put_price)/n_simulations,2),
        'Optimal_stopping_average_returns_calls': round(100*np.sum(max_call_prices - Call_price)/n_simulations, 2),
        'Optimal_stopping_average_returns_puts': round(100*np.sum(max_put_prices - Put_price)/n_simulations, 2)
    }

    return base64.b64encode(image_png).decode('utf-8'), metrics

def home(request):
    if request.method == 'POST':
        form = HeatmapForm(request.POST)
        if form.is_valid():
            stock_price = form.cleaned_data['stock_price']
            strike_price = form.cleaned_data['strike_price']
            time = float(form.cleaned_data['time']/365)
            risk_free_interest_rate = form.cleaned_data['risk_free_interest_rate']
            volatility = form.cleaned_data['volatility']
            n_simulations = int(form.cleaned_data['n_simulations'])
            if n_simulations <= 0:
                n_simulations = 1
            elif n_simulations > 10000:
                n_simulations = 10000
            heatmap_image, metrics = generate_Outputs(stock_price, strike_price, time, risk_free_interest_rate, volatility, n_simulations)
            return JsonResponse({'heatmap_image': heatmap_image, 'metrics': metrics})
    else:
        form = HeatmapForm()
    return render(request, 'home.html', {'form': form})
