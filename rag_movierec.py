import os
import pandas as pd
from sklearn.model_selection import train_test_split
from surprise import Dataset, Reader, KNNBasic
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

# Step 1: Load and preprocess the MovieLens 100K dataset
def load_data():
    data_path = 'ml-100k/u.data'
    column_names = ['user_id', 'item_id', 'rating', 'timestamp']
    df = pd.read_csv(data_path, sep='\t', names=column_names)
    df = df.drop(columns=['timestamp'])
    return df

# Step 2: Split data into train and test sets
def split_data(df):
    train_data, test_data = train_test_split(df, test_size=0.2, random_state=42)
    return train_data, test_data

# Step 3: Prepare data for collaborative filtering using Surprise
def prepare_data_for_surprise(train_data):
    reader = Reader(rating_scale=(1, 5))
    trainset = Dataset.load_from_df(train_data[['user_id', 'item_id', 'rating']], reader).build_full_trainset()
    return trainset

# Step 4: Train the KNN-based collaborative filtering model
def train_knn_model(trainset):
    knn = KNNBasic(sim_options={'name': 'cosine', 'user_based': True})
    knn.fit(trainset)
    return knn

# Step 5: Retrieve top N recommendations for a given user
def get_knn_recommendations(knn, trainset, user_id, k=5):
    inner_id = trainset.to_inner_uid(user_id)
    neighbors = knn.get_neighbors(inner_id, k=k)
    recommendations = []
    for neighbor in neighbors:
        recommendations.extend([item for item, rating in trainset.ur[neighbor]])
    return list(set(recommendations))[:k]

# Step 6: Load pre-trained GPT-2 model and tokenizer
def load_gpt2_model():
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    return model, tokenizer

# Step 7: Generate personalized movie suggestions
def generate_movie_suggestions(model, tokenizer, retrieved_items, user_profile):
    prompt = f"As a fan of {user_profile}, you might enjoy the following movies: "
    for item_id in retrieved_items:
        prompt += f"Movie {item_id}, "
    inputs = tokenizer.encode(prompt, return_tensors='pt')
    outputs = model.generate(inputs, max_length=100, num_return_sequences=1)
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text

# Step 8: Combine retrieval and generation for final recommendations
def get_combined_recommendations(knn, trainset, model, tokenizer, user_id, user_profile, k=5):
    retrieved_items = get_knn_recommendations(knn, trainset, user_id, k)
    augmented_text = generate_movie_suggestions(model, tokenizer, retrieved_items, user_profile)
    return retrieved_items, augmented_text

# Step 9: Main execution
if __name__ == "__main__":
    df = load_data()
    train_data, _ = split_data(df)
    trainset = prepare_data_for_surprise(train_data)
    
    knn = train_knn_model(trainset)
    model, tokenizer = load_gpt2_model()

    user_id = 196
    user_profile = "action movies"
    recommendations, augmented_text = get_combined_recommendations(knn, trainset, model, tokenizer, user_id, user_profile)

    print("Retrieved Items:", recommendations)
    print("Augmented Text:", augmented_text)
