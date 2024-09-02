# Implement a linear regression model to predict the prices of houses based on their square footage and the number of bedrooms and bathrooms.

# Dataset : - https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data


# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load the actual training dataset
train_file_path = 'train.csv'  
train_data = pd.read_csv(train_file_path)

# Selecting relevant columns
selected_data = train_data[['GrLivArea', 'BedroomAbvGr', 'FullBath', 'SalePrice']]

# Checking for missing values in the selected columns
print("Missing values in each column before cleaning:")
print(selected_data.isnull().sum())

# Dropping rows with any missing data in the selected columns
selected_data = selected_data.dropna()

# Splitting the data into features (X) and target variable (y)
X = selected_data[['GrLivArea', 'BedroomAbvGr', 'FullBath']]
y = selected_data['SalePrice']

# Split the dataset into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Output the results
print(f"Mean Squared Error (MSE): {mse}")
print(f"R² Score: {r2}")

# Optional: Visualize the results
import matplotlib.pyplot as plt

plt.scatter(y_test, y_pred)
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.title('Actual vs. Predicted House Prices')
plt.show()
