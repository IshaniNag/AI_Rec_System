import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# Load the dataset
df = pd.read_csv('PC_Components_Dataset.csv')

# Handle missing values
print("Missing values before handling:")
print(df.isnull().sum())

# Optional: You can drop rows with missing values or impute them
# df.dropna(inplace=True)  # To drop rows with missing values
# OR
# df.fillna(method='ffill', inplace=True)  # To forward fill missing values

print("Missing values after handling:")
print(df.isnull().sum())

# Check if you need to scale the numerical features
scaler = MinMaxScaler()

# Scale the 'Price ($)' and 'Rating' columns
df[['Price ($)', 'Rating']] = scaler.fit_transform(df[['Price ($)', 'Rating']])

# Check the processed DataFrame
print("Processed Data:")
print(df.head())

# Optionally, you can split the data into training and testing sets
# We assume the target variable is 'Compatibility' in this case
X = df.drop(columns=['Component Name', 'Compatibility'])  # Features (exclude target)
y = df['Compatibility']  # Target variable

# Split the data into training and test sets (80-20 split)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Save the processed data to a new CSV file
df.to_csv('processed_PC_Components.csv', index=False)

# Optionally, you can save the split training and testing data
X_train.to_csv('X_train.csv', index=False)
X_test.to_csv('X_test.csv', index=False)
y_train.to_csv('y_train.csv', index=False)
y_test.to_csv('y_test.csv', index=False)

# Output the dataset shapes
print(f"Training data shape: {X_train.shape}")
print(f"Test data shape: {X_test.shape}")
