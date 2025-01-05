import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

# Load the preprocessed dataset
df = pd.read_csv('PC_Components_Dataset.csv')

# Display the columns to understand which features are available
print("Columns in dataset:", df.columns)

# Select numerical columns for similarity calculation
numerical_features = ['Price ($)', 'Rating']
categorical_features = ['Category_GPU', 'Category_Monitor', 'Category_Motherboard', 'Category_RAM', 'Category_Storage']

# Standardize the numerical features (important for distance-based models like cosine similarity)
scaler = StandardScaler()
scaled_numerical_features = scaler.fit_transform(df[numerical_features])

# Combine the numerical and categorical features
X = pd.concat([pd.DataFrame(scaled_numerical_features, columns=numerical_features), df[categorical_features]], axis=1)

# Compute the cosine similarity matrix
cosine_sim = cosine_similarity(X)

# Create a DataFrame with the similarity matrix for better understanding
cosine_sim_df = pd.DataFrame(cosine_sim, index=df['Component Name'], columns=df['Component Name'])

# Function to get recommendations based on the similarity of components
def get_recommendations(component_name, top_n=5):
    # Get the pairwise similarity scores for the given component
    sim_scores = cosine_sim_df[component_name]
    
    # Sort the scores in descending order and exclude the first entry (the component itself)
    sim_scores = sim_scores.sort_values(ascending=False)[1:]
    
    # Get the top N most similar components
    top_similar = sim_scores.head(top_n)
    
    return top_similar.index.tolist()

# Test the function with a component name
component_name = 'AMD Ryzen 9 5950X'
recommended_components = get_recommendations(component_name)

# Display recommended components
print(f"Recommended components for {component_name}:")
for component in recommended_components:
    print(component)
