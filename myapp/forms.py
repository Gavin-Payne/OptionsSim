from django import forms

class HeatmapForm(forms.Form):
    stock_price = forms.FloatField(label='Stock Price', initial=100)
    strike_price = forms.FloatField(label='Strike Price', initial=100)
    time = forms.FloatField(label='Time (days)', initial=60)
    risk_free_interest_rate = forms.FloatField(label='Risk Free Interest Rate', initial=0.0428)
    volatility = forms.FloatField(label='Volatility', initial=0.2)
    n_simulations = forms.FloatField(label='Number of Sims', initial=200)
    
    

