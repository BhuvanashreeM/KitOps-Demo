import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

base_dir = './'

# Define subdirectories
directories = [
    os.path.join(base_dir, 'models', 'model_snapshot1'),
    os.path.join(base_dir, 'models', 'model_snapshot2'),
    os.path.join(base_dir, 'datasets'),
]

# Create directories
for directory in directories:
    os.makedirs(directory, exist_ok=True)
    print(f"Created directory: {directory}")

print("Directory structure created successfully!")


# Create Movie Streaming Data Snapshots
def generate_movie_streaming_data(num_users=1000, num_movies=200, days=30, seed=42):
    """
    Generate synthetic movie streaming data
    
    Parameters:
    -----------
    num_users: int, number of unique users
    num_movies: int, number of unique movies
    days: int, number of days to generate data for
    seed: int, random seed for reproducibility
    
    Returns:
    --------
    pandas DataFrame with streaming data
    """
    np.random.seed(seed)
    
    user_ids = [f"user_{i}" for i in range(1, num_users + 1)]
    movie_ids = [f"movie_{i}" for i in range(1, num_movies + 1)]
    genres = ['Action', 'Comedy', 'Drama', 'SciFi', 'Romance', 'Thriller', 'Horror', 'Documentary']
    
    movie_metadata = pd.DataFrame({
        'movie_id': movie_ids,
        'genre': np.random.choice(genres, num_movies),
        'release_year': np.random.randint(1980, 2023, num_movies),
        'duration_minutes': np.random.randint(70, 210, num_movies),
        'popularity_score': np.random.uniform(1, 10, num_movies).round(1)
    })
    
    # Generate streaming events
    start_date = datetime.now() - timedelta(days=days)
    data = []
    
    # Each user watches between 1 and 15 movies in this period
    for user_id in user_ids:
        age = np.random.randint(18, 70)
        gender = np.random.choice(['M', 'F', 'Other'])
        subscription_type = np.random.choice(['Basic', 'Standard', 'Premium'])
        
        num_watches = np.random.randint(1, 16)
        watched_movies = np.random.choice(movie_ids, num_watches)
        
        for movie_id in watched_movies:
            movie_meta = movie_metadata[movie_metadata['movie_id'] == movie_id].iloc[0]
            stream_date = start_date + timedelta(days=np.random.randint(0, days))
            stream_hour = np.random.randint(0, 24)
            stream_datetime = stream_date.replace(hour=stream_hour)
            
            watch_percentage = np.random.uniform(0.1, 1.0)
            watch_time = int(movie_meta['duration_minutes'] * watch_percentage)
            
            completed = 1 if watch_percentage > 0.9 else 0
            
            device = np.random.choice(['Mobile', 'Tablet', 'Smart TV', 'Computer', 'Game Console'])
            
            rating = np.random.randint(1, 6) if np.random.random() < 0.3 else np.nan
            
            # Add data
            data.append({
                'user_id': user_id,
                'movie_id': movie_id,
                'timestamp': stream_datetime,
                'watch_time_minutes': watch_time,
                'completed': completed,
                'device': device,
                'rating': rating,
                'user_age': age,
                'user_gender': gender,
                'subscription_type': subscription_type,
                'genre': movie_meta['genre'],
                'release_year': movie_meta['release_year'],
                'movie_duration': movie_meta['duration_minutes'],
                'popularity_score': movie_meta['popularity_score']
            })
    
    df = pd.DataFrame(data)
    return df

# Generate first snapshot
print("Generating first data snapshot...")
df1 = generate_movie_streaming_data(num_users=1000, num_movies=200, days=30, seed=42)

# Generate second snapshot (more recent data with additional users)
print("Generating second data snapshot...")
df2 = generate_movie_streaming_data(num_users=1200, num_movies=250, days=30, seed=43)

# Save datasets
os.makedirs('./datasets', exist_ok=True)
df1.to_csv('./datasets/streaming_data_snapshot1.csv', index=False)
df2.to_csv('./datasets/streaming_data_snapshot2.csv', index=False)

print(f"Saved streaming_data_snapshot1.csv with {len(df1)} records")
print(f"Saved streaming_data_snapshot2.csv with {len(df2)} records")