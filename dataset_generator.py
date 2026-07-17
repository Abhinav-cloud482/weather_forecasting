import pandas as pd
import numpy as np

# For reproducibility
np.random.seed(42)

rows = 1000

days = np.arange(1, rows + 1)

# Generate realistic weather features
temperature = np.random.normal(30, 5, rows)          # Average 30°C
temperature = np.clip(temperature, 15, 45)

humidity = np.random.normal(65, 12, rows)
humidity = np.clip(humidity, 30, 100)

wind_speed = np.random.normal(12, 4, rows)
wind_speed = np.clip(wind_speed, 0, 30)

precipitation = np.random.binomial(1, 0.30, rows)

# Create target (Next Day Temperature)
next_day_temperature = (
    temperature
    + np.random.normal(0, 1.5, rows)
    - precipitation * 1.2
    - (humidity - 65) * 0.03
    + (wind_speed - 12) * 0.05
)

next_day_temperature = np.clip(next_day_temperature, 15, 45)

# Create DataFrame
df = pd.DataFrame({
    "Day": days,
    "Temperature": np.round(temperature, 1),
    "Humidity": np.round(humidity, 1),
    "Wind Speed": np.round(wind_speed, 1),
    "Precipitation": precipitation,
    "Next Day Temperature": np.round(next_day_temperature, 1)
})

print(df.head())

# Save to CSV
df.to_csv("weather_data_1000.csv", index=False)

print("\nDataset saved as weather_data_1000.csv")
print(f"Total rows: {len(df)}")