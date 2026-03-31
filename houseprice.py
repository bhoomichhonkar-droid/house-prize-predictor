import pandas as pd
from sklearn.linear_model import LinearRegression
import pickle
# Load CSV Data

df = pd.read_csv("Bengaluru_House_Data.csv")
# Clean theData
df = df.dropna()

# Convert total_sqft to numeric
df['total_sqft'] = pd.to_numeric(df['total_sqft'], errors='coerce')
df = df.dropna()

# Select Features and assign the features which are in your csv file

print(df.columns)   # run once to see column names

X = df[['total_sqft', 'bath']]   # change if needed
y = df['price']    

# Train the model
model = LinearRegression()
model.fit(X, y)

# Step 5: Predict 
print("=== House Price Predictor ===")
area = int(input("Enter area: "))
bedrooms = int(input("Enter bedrooms: "))

price = model.predict([[area, bedrooms]])
print("Predicted Price:", round(price[0],2))

# Save Model (optional)
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model saved as model.pkl")
