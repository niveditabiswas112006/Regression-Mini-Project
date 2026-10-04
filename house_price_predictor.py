import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# 1. Generate a completely unique synthetic dataset to avoid plagiarism
np.random.seed(123) # Using a specific seed so it's consistent but unique
num_samples = 1250

square_feet = np.random.randint(800, 4500, size=num_samples)
bedrooms = np.random.randint(1, 6, size=num_samples)
bathrooms = np.random.randint(1, 4, size=num_samples)
year_built = np.random.randint(1960, 2023, size=num_samples)
distance_to_center = np.random.uniform(2.0, 35.0, size=num_samples)

# Generate prices with realistic coefficients and random noise
noise = np.random.normal(0, 30000, size=num_samples)
house_prices = (
    180 * square_feet + 
    15000 * bedrooms + 
    20000 * bathrooms - 
    900 * (2024 - year_built) - 
    1800 * distance_to_center + 
    60000 + noise
)

# Create DataFrame
df = pd.DataFrame({
    'SquareFeet': square_feet,
    'Bedrooms': bedrooms,
    'Bathrooms': bathrooms,
    'YearBuilt': year_built,
    'DistanceToCenter_km': distance_to_center,
    'HousePrice': house_prices
})

# 2. Save the dataset to a CSV file (Deliverable)
csv_filename = "unique_house_prices_dataset.csv"
df.to_csv(csv_filename, index=False)
print(f"Unique synthetic dataset saved to {csv_filename}")

# 3. Split the data
X = df.drop('HousePrice', axis=1)
y = df['HousePrice']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

# 4. Train the Linear Regression model
print("Training Linear Regression model...")
model = LinearRegression()
model.fit(X_train, y_train)

# 5. Make predictions
y_pred = model.predict(X_test)

# 6. Evaluate the model (Deliverable)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("-" * 30)
print("Model Evaluation:")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"R-squared (R2): {r2:.4f}")
print("-" * 30)

# 7. Visualize actual vs. predicted values (Deliverable)
plt.figure(figsize=(9, 6))
plt.scatter(y_test, y_pred, alpha=0.7, color='#2ca02c', edgecolor='white', s=60) # Customized appearance
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='#d62728', linestyle='--', lw=2.5)
plt.title('Actual vs. Predicted House Prices (Custom Generated Data)', fontsize=14, pad=15)
plt.xlabel('Actual House Prices ($)', fontsize=12)
plt.ylabel('Predicted House Prices ($)', fontsize=12)
plt.grid(True, linestyle=':', alpha=0.7)
plt.tight_layout()

# Save the scatter plot
plot_filename = "actual_vs_predicted_unique.png"
plt.savefig(plot_filename)
print(f"Scatter plot saved to {plot_filename}")
