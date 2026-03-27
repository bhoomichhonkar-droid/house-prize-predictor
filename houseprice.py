import pandas as pd
from sklearn.linear_model import LinearRegression
import pickle
# Step 1: Load CSV Data
# -----------------------------
df = pd.read_csv("Bengaluru_House_Data.csv")
# Step 2: Clean Data
# -----------------------------
df = df.dropna()

# Convert total_sqft to numeric
df['total_sqft'] = pd.to_numeric(df['total_sqft'], errors='coerce')
df = df.dropna()


# Step 3: Select Features
# (IMPORTANT: change column names according to your dataset)
# -----------------------------
print(df.columns)   # run once to see column names

X = df[['total_sqft', 'bath']]   # change if needed
y = df['price']    

# Step 4: Train
model = LinearRegression()
model.fit(X, y)

# Step 5: Predict
print("=== House Price Predictor ===")
area = int(input("Enter area: "))
bedrooms = int(input("Enter bedrooms: "))

price = model.predict([[area, bedrooms]])
print("Predicted Price:", round(price[0],2))
# Step 6: Save Model (optional)
# -----------------------------
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model saved as model.pkl")