import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

# Load dataset
df = pd.read_excel("C:\\Users\\shant\\OneDrive\\Desktop\\gym recommendation\\gym recommendation.xlsx")  # Replace with the actual file path

# Step 1: Data Preprocessing
# Encoding categorical variables
label_encoders = {}
categorical_columns = ['Hypertension', 'Diabetes', 'Level', 'Fitness Goal', 'Fitness Type', 'Exercises', 'Equipment', 'Diet']

for col in categorical_columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le  # Save encoder for inverse transform if needed

# Normalize numerical features
scaler = MinMaxScaler()
df[['Age', 'Height', 'Weight', 'BMI']] = scaler.fit_transform(df[['Age', 'Height', 'Weight', 'BMI']])

# Combine features for recommendation
df['Combined Features'] = (
    df['Fitness Goal'].astype(str) + ' ' +
    df['Fitness Type'].astype(str) + ' ' +
    df['Exercises'].astype(str) + ' ' +
    df['Diet'].astype(str) + ' ' +
    df['Level'].astype(str)
)

# Step 2: Content Representation
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['Combined Features'])

# Step 3: Compute Similarity
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Step 4: Recommendation Function
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
    idx = df.index[df['ID'] == user_id].tolist()[0]  # Get index of the user
    sim_scores = list(enumerate(cosine_sim[idx]))  # Pair each index with similarity score
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)  # Sort by score
    sim_scores = sim_scores[1:top_n+1]  # Exclude self-match

    workout_indices = [i[0] for i in sim_scores]  # Get indices of top workouts
    return df.iloc[workout_indices][['ID', 'Fitness Goal', 'Fitness Type', 'Exercises', 'Diet', 'Recommendation']]

# Step 5: Testing the Recommendation System
#user_id_to_test = 12345  # Replace with an actual ID from the dataset
#recommendations = get_recommendations(user_id_to_test, top_n=5)
#print("Top Recommendations:")
#print(recommendations)

# Optionally, save recommendations
#recommendations.to_csv('user_recommendations.csv', index=False)

import pickle

# Save the DataFrame and cosine similarity matrix
with open('workout_data.pkl', 'wb') as file:
    pickle.dump((df, cosine_sim), file)
