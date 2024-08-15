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
