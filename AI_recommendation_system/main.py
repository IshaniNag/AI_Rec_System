import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import LabelEncoder

# Load the processed dataset
df = pd.read_csv('AI_recommendation_system/processed_PC_Components.csv')

# Initialize LabelEncoder for categorical columns
label_encoder = LabelEncoder()

# Encode categorical columns: 'Performance', 'Purpose', 'Category_GPU', 'Category_RAM', etc.
df['Performance_encoded'] = label_encoder.fit_transform(df['Performance'])
df['Purpose_encoded'] = label_encoder.fit_transform(df['Purpose'])

# Function to calculate content-based recommendations
def recommend_components(purpose, budget):
    # Encode the user-selected purpose using the same label encoder
    purpose_label = label_encoder.transform([purpose])[0]
    
    # Filter the dataset based on the user's budget and selected purpose
    filtered_df = df[(df['Purpose_encoded'] == purpose_label) & (df['Price ($)'] <= budget)]
    
    if filtered_df.empty:
        return "No components found within the given budget and purpose."
    
    # Prepare the features for similarity computation (drop non-numeric columns)
    feature_columns = ['Price ($)', 'Rating', 'Performance_encoded']
    filtered_df_features = filtered_df[feature_columns]
    
    # Compute similarity of all filtered components to each other
    similarities = cosine_similarity(filtered_df_features)
    
    # Get the indices of the most similar components (exclude the first one as it is the selected component)
    similar_indices = similarities.argsort().flatten()[-6:-1]
    
    # Get the names of the recommended components
    recommended_components = filtered_df.iloc[similar_indices]
    
    # Group by each category (GPU, RAM, Storage, Monitor, etc.)
    categories = ['Category_GPU', 'Category_Motherboard', 'Category_RAM', 'Category_Storage', 'Category_Monitor']
    recommended_by_category = {}
    
    for category in categories:
        category_df = recommended_components[recommended_components[category] == 1]
        
        if not category_df.empty:
            recommended_by_category[category] = category_df.iloc[0]  # Taking the first one from the category
    
    # If a category is empty, it means no component from that category was found
    recommended_output = []
    
    # Ensure that we always show at least one monitor
    if 'Category_Monitor' not in recommended_by_category:
        monitor_df = df[df['Category_Monitor'] == 1]  # Get all monitors from the dataset
        if not monitor_df.empty:
            recommended_by_category['Category_Monitor'] = monitor_df.iloc[0]  # Show the first available monitor
    
    # Ensure that we always show at least one storage component
    if 'Category_Storage' not in recommended_by_category:
        storage_df = df[df['Category_Storage'] == 1]  # Get all storage components from the dataset
        if not storage_df.empty:
            recommended_by_category['Category_Storage'] = storage_df.iloc[0]  # Show the first available storage component
    
    for category in categories:
        if category in recommended_by_category:
            component = recommended_by_category[category]
            recommended_output.append(f"Component: {component['Component Name']} | Type: {category.split('_')[1]}")
        else:
            recommended_output.append(f"No {category.split('_')[1]} found | Type: {category.split('_')[1]}")
    
    return recommended_output

# Streamlit UI
st.title("PC Component Recommendation System")

# Dropdown for selecting a purpose (use original Purpose values from the dataset)
purpose_options = df['Purpose'].unique()  # Extract unique purposes from the dataset
purpose = st.selectbox("Select a Purpose", purpose_options)  # User-friendly dropdown options

# Input for budget
budget = st.number_input("Enter Your Budget ($)", min_value=0, max_value=5000, value=1000)

# Button to get recommendations
if st.button("Get Recommendations"):
    recommendations = recommend_components(purpose, budget)
    st.write("Recommended Components:")
    
    if isinstance(recommendations, str):
        st.write(recommendations)  # Show the message if no recommendations
    else:
        for comp in recommendations:
            st.write(comp)
