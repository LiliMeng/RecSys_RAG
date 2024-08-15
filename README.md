# RecSys_RAG

Creating a Retrieved Augmented Generation (RAG) system using the MovieLens 100K dataset involves integrating both retrieval-based methods and generative models to create a recommendation engine. Below is a simplified example of how this could be implemented using Python and relevant libraries such as PyTorch, Hugging Face Transformers, and some basic collaborative filtering techniques.

### Step 1: Setup and Data Loading

First, let's load the MovieLens 100K dataset and preprocess it for both retrieval and generation.

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from surprise import Dataset, Reader, KNNBasic
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

# Load MovieLens 100K dataset
data_path = 'ml-100k/u.data'
column_names = ['user_id', 'item_id', 'rating', 'timestamp']
df = pd.read_csv(data_path, sep='\t', names=column_names)

# Drop timestamp
df = df.drop(columns=['timestamp'])

# Split data into train and test sets
train_data, test_data = train_test_split(df, test_size=0.2, random_state=42)

# Load the dataset into Surprise for collaborative filtering
reader = Reader(rating_scale=(1, 5))
trainset = Dataset.load_from_df(train_data[['user_id', 'item_id', 'rating']], reader).build_full_trainset()
testset = Dataset.load_from_df(test_data[['user_id', 'item_id', 'rating']], reader).build_full_trainset().build_testset()

# Use KNNBasic algorithm from Surprise as the retrieval method
knn = KNNBasic(sim_options={'name': 'cosine', 'user_based': True})
knn.fit(trainset)
```

### Step 2: Retrieval - Collaborative Filtering

Using a simple collaborative filtering method like KNN to retrieve a set of similar items or users.

```python
# Function to retrieve top N recommendations for a given user
def get_knn_recommendations(user_id, k=5):
    inner_id = trainset.to_inner_uid(user_id)
    neighbors = knn.get_neighbors(inner_id, k=k)
    recommendations = []
    for neighbor in neighbors:
        recommendations.extend([item for item, rating in trainset.ur[neighbor]])
    return list(set(recommendations))[:k]

# Example of getting recommendations for a user
user_id = 196
retrieved_items = get_knn_recommendations(user_id)
print("Retrieved Items:", retrieved_items)
```

### Step 3: Generative Augmentation

Now, we integrate a generative model to augment the retrieved recommendations with personalized suggestions or generated text.

```python
# Load pre-trained GPT-2 model and tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")

# Function to generate personalized movie suggestions
def generate_movie_suggestions(retrieved_items, user_profile):
    prompt = f"As a fan of {user_profile}, you might enjoy the following movies: "
    for item_id in retrieved_items:
        prompt += f"Movie {item_id}, "
    inputs = tokenizer.encode(prompt, return_tensors='pt')
    outputs = model.generate(inputs, max_length=100, num_return_sequences=1)
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text

# Example usage
user_profile = "action movies"
augmented_recommendations = generate_movie_suggestions(retrieved_items, user_profile)
print("Augmented Recommendations:", augmented_recommendations)
```

### Step 4: Combine and Present Recommendations

Finally, you can combine the retrieved and generated recommendations to present them to the user.

```python
def get_combined_recommendations(user_id, user_profile, k=5):
    retrieved_items = get_knn_recommendations(user_id, k)
    augmented_text = generate_movie_suggestions(retrieved_items, user_profile)
    return retrieved_items, augmented_text

# Example for a specific user
user_id = 196
user_profile = "action movies"
recommendations, augmented_text = get_combined_recommendations(user_id, user_profile)

print("Retrieved Items:", recommendations)
print("Augmented Text:", augmented_text)
```

### Explanation:

- **Retrieval**: We use collaborative filtering (KNN) to retrieve relevant items (movies) based on similar users' ratings.
- **Generation**: The GPT-2 model generates personalized recommendations or suggestions based on the retrieved items and a user profile.
- **Combination**: The system combines the retrieved items with the generated suggestions to provide a richer recommendation experience.

### Summary

This example demonstrates how you could implement a basic RAG system for the MovieLens 100K dataset. The retrieval step (collaborative filtering) finds relevant items, and the generative model (GPT-2) augments these with personalized suggestions, making the recommendation system more versatile and engaging.

For more advanced implementations, you could fine-tune the GPT-2 model on movie-related data, incorporate more complex retrieval strategies, and enhance the interaction between the retrieval and generative components.
