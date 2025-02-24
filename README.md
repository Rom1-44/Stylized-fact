import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error

# Generate synthetic data
def generate_synthetic_data(n_samples=1000):
    np.random.seed(42)
    S = np.random.uniform(50, 150, n_samples)  # Stock price
    K = np.random.uniform(50, 150, n_samples)  # Strike price
    T = np.random.uniform(0.1, 2, n_samples)   # Time to maturity
    r = np.random.uniform(0.01, 0.05, n_samples)  # Risk-free rate
    sigma = np.random.uniform(0.1, 0.5, n_samples)  # Volatility

    # Black-Scholes formula for call option price
    def black_scholes_call(S, K, T, r, sigma):
        d1 = (np.log(S / K) + (r + sigma ** 2 / 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        call = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        return call

    call_prices = black_scholes_call(S, K, T, r, sigma)
    data = pd.DataFrame({'S': S, 'K': K, 'T': T, 'r': r, 'sigma': sigma, 'call_price': call_prices})
    return data

# Generate data
data = generate_synthetic_data()

# Split data into features and target
X = data.drop('call_price', axis=1)
y = data['call_price']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize and train the MLPRegressor model
mlp = MLPRegressor(hidden_layer_sizes=(50, 50), max_iter=1000, random_state=42)
mlp.fit(X_train_scaled, y_train)

# Make predictions
y_pred = mlp.predict(X_test_scaled)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

# Example prediction
example = np.array([[100, 100, 1, 0.02, 0.2]])
example_scaled = scaler.transform(example)
predicted_price = mlp.predict(example_scaled)
print(f'Predicted Call Option Price: {predicted_price[0]}')
