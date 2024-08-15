# RecSys_RAG
This script loads the MovieLens 100K dataset, applies a collaborative filtering retrieval method, and then uses a generative model to augment the recommendations. 


### How to Execute the Script

1. **Install Required Libraries**:
   - You’ll need to install the required Python libraries. You can do this using pip:
     ```bash
     pip install pandas scikit-learn surprise transformers torch
     ```

2. **Download the MovieLens 100K Dataset**:
   - Ensure that you have the MovieLens 100K dataset in your working directory under the path `ml-100k/u.data`. You can download it from [MovieLens](https://grouplens.org/datasets/movielens/100k/).

3. **Run the Script**:
   - Save the script above as `rag_movierec.py`.
   - Run it using Python:
     ```bash
     python rag_movierec.py
     ```

### Explanation:

- **Data Loading and Preprocessing**: The script first loads and preprocesses the MovieLens 100K data.
- **Collaborative Filtering**: A KNN-based collaborative filtering model is trained on the dataset to retrieve recommendations based on similar users.
- **Generative Model**: GPT-2 is used to augment these recommendations with personalized text based on the retrieved items and a user profile.
- **Combined Recommendations**: Finally, the script combines the retrieval and generation steps to produce a set of movie recommendations and personalized suggestions.

This script provides a basic implementation of RAG for a recommendation system, which you can expand and refine according to your specific requirements.


Let's break down the function `generate_movie_suggestions` step by step:

### Purpose:
The function `generate_movie_suggestions` generates personalized movie suggestions using a pre-trained generative language model (like GPT-2). It combines the user's profile and a list of retrieved movie items to produce a coherent text that might suggest movies the user would enjoy.

### Function Definition:
```python
def generate_movie_suggestions(model, tokenizer, retrieved_items, user_profile):
    prompt = f"As a fan of {user_profile}, you might enjoy the following movies: "
    for item_id in retrieved_items:
        prompt += f"Movie {item_id}, "
    inputs = tokenizer.encode(prompt, return_tensors='pt')
    outputs = model.generate(inputs, max_length=100, num_return_sequences=1)
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text
```

### Detailed Explanation:

1. **Input Parameters:**
   - **`model`**: This is the pre-trained generative model, typically a language model like GPT-2, which can generate human-like text.
   - **`tokenizer`**: The tokenizer is responsible for converting the input text into tokens that the model can understand. It also converts the model's output back into readable text.
   - **`retrieved_items`**: A list of movie IDs that have been retrieved by the collaborative filtering model. These are the movies that the system has identified as potentially interesting to the user.
   - **`user_profile`**: A description of the user's preferences, such as "action movies" or "romantic comedies," which helps tailor the generated text to the user's tastes.

2. **Prompt Construction:**
   ```python
   prompt = f"As a fan of {user_profile}, you might enjoy the following movies: "
   for item_id in retrieved_items:
       prompt += f"Movie {item_id}, "
   ```
   - The function starts by creating a `prompt` string that forms the initial input for the generative model.
   - The prompt begins with a template phrase: `"As a fan of {user_profile}, you might enjoy the following movies: "`.
   - The function then loops through each `item_id` in `retrieved_items`, appending each movie ID to the prompt. This prompt now contains the user's profile description and a list of movies that the collaborative filtering model retrieved.

3. **Tokenization:**
   ```python
   inputs = tokenizer.encode(prompt, return_tensors='pt')
   ```
   - The prompt string is tokenized using the `tokenizer.encode` method. The `return_tensors='pt'` argument specifies that the output should be a PyTorch tensor, which is the format required by the GPT-2 model.

4. **Text Generation:**
   ```python
   outputs = model.generate(inputs, max_length=100, num_return_sequences=1)
   ```
   - The `model.generate` method generates text based on the input tokens. 
   - `max_length=100` limits the length of the generated text to 100 tokens.
   - `num_return_sequences=1` specifies that only one sequence of text should be generated.

5. **Decoding:**
   ```python
   generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
   ```
   - The generated tokens are decoded back into human-readable text using the `tokenizer.decode` method.
   - `skip_special_tokens=True` ensures that any special tokens used by the model (like padding or start/end tokens) are not included in the final output.

6. **Return Statement:**
   ```python
   return generated_text
   ```
   - The final generated text, which contains the personalized movie suggestions, is returned as the output of the function.

### Example:

Given a user who likes "action movies" and a set of retrieved movie IDs `[123, 456, 789]`, the function might generate a suggestion like:

```plaintext
"As a fan of action movies, you might enjoy the following movies: Movie 123, Movie 456, Movie 789. These films offer thrilling action scenes, gripping storylines, and heroic characters that will keep you on the edge of your seat."
```

### Summary:
- **Combining Retrieval and Generation**: The function first uses collaborative filtering to retrieve relevant movies and then uses a generative model to create a personalized recommendation text.
- **Application**: This approach can be particularly useful in recommendation systems where generating personalized, engaging text can enhance the user experience. The combination of structured data (retrieved items) and unstructured data (generated text) can lead to a richer, more satisfying recommendation.
