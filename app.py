import streamlit as st
import pandas as pd
import pickle
import numpy as np  # Ensure numpy is imported for cosine similarity

# Load the saved DataFrame and label encoders
with open('workout_data.pkl', 'rb') as file:
    df, label_encoders = pickle.load(file)

# Placeholder cosine similarity matrix (replace with actual precomputed matrix)
cosine_sim = np.random.rand(len(df), len(df))  # Example matrix; replace with your actual similarity computation.

def reverse_label_encoding(df, label_encoders, columns):
    """
    Converts encoded values back to their original labels.
    Args:
        df: DataFrame containing encoded data.
        label_encoders: Dictionary of LabelEncoders for each column.
        columns: List of columns to decode.

    Returns:
        DataFrame with decoded columns.
    """
    for col in columns:
        if col in label_encoders:  # Check if encoder exists
            df[col] = df[col].map(lambda x: label_encoders[col].inverse_transform([x])[0] if pd.notna(x) else x)
        else:
            raise KeyError(f"Label encoder for column '{col}' not found.")
    return df

# Define the recommendation function
def get_recommendations(user_id, cosine_sim, df, top_n=5):
    """
    Recommend workout plans based on user's ID.
    Args:
        user_id: ID of the user.
        cosine_sim: Precomputed cosine similarity matrix.
        df: DataFrame containing workout data.
        top_n: Number of recommendations to return.

    Returns:
        DataFrame containing top workout recommendations.
    """
    try:
        idx = df.index[df['ID'] == user_id].tolist()[0]  # Get index of the user
        sim_scores = list(enumerate(cosine_sim[idx]))  # Pair each index with similarity score
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)  # Sort by score
        sim_scores = sim_scores[1:top_n+1]  # Exclude self-match

        workout_indices = [i[0] for i in sim_scores]  # Get indices of top workouts
        return df.iloc[workout_indices][['ID', 'Fitness Goal', 'Fitness Type', 'Exercises', 'Diet', 'Recommendation']]
    except IndexError:
        return pd.DataFrame()  # Return empty DataFrame if user ID not found

# Streamlit App UI
st.set_page_config(page_title="Gym Recommendation System", layout="wide", page_icon="🏋️‍♂️")

st.title("🏋️‍♂️ Personalized Workout Plan Recommendation System")
st.markdown("Discover tailored workout plans and diets based on your goals!")

# Sidebar for user input
st.sidebar.header("Your Details")
# Input Member ID
user_id = st.sidebar.number_input(
    "Enter your Member ID (e.g., 101, 102):", 
    min_value=1, 
    step=1, 
    help="Find your Member ID in your gym registration details."
)

# Validate Member ID
if user_id in df['ID'].values:
    st.sidebar.success("Member ID found!")

    # Get user's details for context
    user_details = df[df['ID'] == user_id].iloc[0]

    # Decode user's Fitness Goal and Fitness Type
    decoded_details = reverse_label_encoding(user_details.to_frame().T, label_encoders, ['Fitness Goal', 'Fitness Type'])
    decoded_details = decoded_details.iloc[0]

    # Display the decoded details
    st.sidebar.write(f"🎯 Fitness Goal: {decoded_details['Fitness Goal']}")
    st.sidebar.write(f"💪 Fitness Type: {decoded_details['Fitness Type']}")
else:
    st.sidebar.error("Invalid Member ID. Please check again.")


top_n = st.sidebar.slider("How many recommendations would you like?", min_value=1, max_value=10, value=5, help="Select the number of recommendations you want.")

# Main content
if st.sidebar.button("Get Recommendations"):
    recommendations = get_recommendations(user_id, cosine_sim, df, top_n=top_n)

    if not recommendations.empty:
        # Convert encoded values back to original labels
        label_columns = ['Fitness Goal', 'Fitness Type', 'Exercises', 'Diet']
        recommendations = reverse_label_encoding(recommendations, label_encoders, label_columns)

        st.header(f"Top {top_n} Workout Recommendations for Member ID: {user_id}")

        for idx, row in recommendations.iterrows():
            with st.expander(f"Recommendation #{idx + 1}"):
                st.write(f"**🏆 Fitness Goal:** {row['Fitness Goal']}")
                st.write(f"**🧘‍♂️ Fitness Type:** {row['Fitness Type']}")
                st.write(f"**💪 Exercises:** {row['Exercises']}")
                st.write(f"**🍎 Diet Plan:** {row['Diet']}")
                st.write(f"**📜 Additional Notes:** {row['Recommendation']}")
                st.markdown("---")
    else:
        st.error("No recommendations found for this Member ID. Please try a different ID or contact support.")

# Feedback Section
st.sidebar.markdown("### Your Feedback")
feedback = st.sidebar.text_area("What do you think about these recommendations?")
if st.sidebar.button("Submit Feedback"):
    st.sidebar.success("Thank you for your feedback!")

# Footer
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown(
    "Developed by [Navashanthan](https://github.com/IT22274984/Gym_Recommendation_System.git). "
    "Your personalized workout planner!"
)
