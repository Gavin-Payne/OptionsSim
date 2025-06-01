# OptionsSim

OptionsSim is a simple Django web application designed to simulate and visualize options pricing and trading scenarios. The project allows users to input parameters such as stock price, strike price, time to expiration, risk-free interest rate, volatility, and the number of simulations. It then generates heatmaps and metrics to help visualize the outcomes of different option strategies.

## Purpose

This project was created as a personal learning exercise to explore web development with Django and to deepen my understanding of financial markets, particularly options pricing and simulation. It combines my interest in finance with my desire to learn how to build and deploy web applications.

## Features

- Black-Scholes option pricing calculations
- Monte Carlo simulations for option price paths
- Interactive web form for user input
- Visualization of results using heatmaps and metrics
- Simple, Bootstrap-themed UI

## How to Run

1. **Install dependencies**  
   Make sure you have Python 3 and pip installed.  
   Install the required packages:
   ```
   pip install -r requirements.txt
   ```

2. **Run migrations**  
   ```
   python manage.py migrate
   ```

3. **Start the development server**  
   ```
   python manage.py runserver
   ```

## Disclaimer

This project is not very well made, but we all start somewhere! It's a good memory of my journey learning Django, web development, and more about financial markets. The code is messy, the UI is basic, and there are probably bugs—but that's all part of the process.
