import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv("temprature.csv")

print("Historical Weather Data:")
print(df.head())

# Define features and target
X = df[['Temperature', 'Humidity', 'Wind Speed', 'Precipitation']]
y = df['Next Day Temperature']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"\nMean Squared Error: {mse:.2f}")
print(f"R^2 Score: {r2:.2f}")

results = pd.DataFrame({
    "Actual": y_test.values,
    "Predicted": y_pred
})

print("\nActual vs Predicted:")
print(results)

plt.figure(figsize=(10, 6))
plt.plot(y_test.values, label="Actual Temperatures", marker='o')
plt.plot(y_pred, label="Predicted Temperatures", marker='x')
plt.title("Actual vs Predicted Temperatures")
plt.xlabel("Test Sample Index")
plt.ylabel("Temperature (°C)")
plt.legend()
plt.grid(True)
plt.show()

# Predict the next day's temperature
new_data = pd.DataFrame({
    "Temperature": [30],
    "Humidity": [60],
    "Wind Speed": [10],
    "Precipitation": [0]
})

predicted_temperature = model.predict(new_data)

print(f"\nPredicted temperature for the next day: {predicted_temperature[0]:.2f}°C")