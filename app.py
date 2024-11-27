import streamlit as st
import pandas as pd
import pickle

# Load the saved data (DataFrame and cosine similarity matrix)
with open('workout_data.pkl', 'rb') as file:
    df, cosine_sim = pickle.load(file)

# Define the recommendation function
def get_recommendations(user_id, cosine_sim=cosine_sim, df=df, top_n=5):
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
st.title("🏋️‍♂️ Personalized Workout Plan Recommendation System")

# Sidebar for user input
st.sidebar.header("Enter Your Details")
user_id = st.sidebar.number_input("Enter your User ID:", min_value=1, step=1)

# Main content
if st.sidebar.button("Get Recommendations"):
    recommendations = get_recommendations(user_id)

    if not recommendations.empty:
        st.header(f"Top Workout Recommendations for User ID: {user_id}")
        for idx, row in recommendations.iterrows():
            st.subheader(f"Recommendation #{idx + 1}")
            st.write(f"**Fitness Goal:** {row['Fitness Goal']}")
            st.write(f"**Fitness Type:** {row['Fitness Type']}")
            st.write(f"**Exercises:** {row['Exercises']}")
            st.write(f"**Diet:** {row['Diet']}")
            st.write(f"**Recommendation:** {row['Recommendation']}")
            st.markdown("---")
    else:
        st.error("No recommendations found for this User ID. Please try a different ID.")

# Footer
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("Developed by [Navashanthan ](https://github.com/IT22274984/Gym_Recommendation_System.git)")

