import pandas as pd
import numpy as np
import faiss
import os
import pickle
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Tuple, Any

class LocationEmbedder:
    def __init__(self, model_name: str = "all-mpnet-base-v2"):
        """
        Initialize the LocationEmbedder with a Sentence Transformer model.
        
        Args:
            model_name: The name of the sentence-transformers model to use
        """
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.locations_df = None
        self.location_texts = None
        self.embedding_dim = None
        
    def load_and_process_data(self, csv_path: str) -> pd.DataFrame:
        """
        Load location data from CSV and process it for embedding.
        
        Args:
            csv_path: Path to the locations.csv file
            
        Returns:
            Processed DataFrame
        """
        # Load the CSV file
        df = pd.read_csv(csv_path)
        
        # Fill NaN values
        df = df.fillna('')
        
        # Create a combined text field for embedding
        df['combined_text'] = df.apply(
            lambda row: f"Title: {row['title']} \n"
                       f"Location: {row['Location / Area']} \n"
                       f"Category: {row['Category / Type']} \n"
                       f"Themes: {row['Theme / Highlights']} \n"
                       f"Price: {row['Price Range']} \n"
                       f"Audience: {row['Audience / Suitability']} \n"
                       f"Hours: {row['Operating Hours']} \n"
                       f"Attributes: {row['Additional Attributes']} \n"
                       f"Content: {row['content']}",
            axis=1
        )
        
        self.locations_df = df
        self.location_texts = df['combined_text'].tolist()
        return df
    
    def create_embeddings(self) -> np.ndarray:
        """
        Create embeddings for all locations.
        
        Returns:
            Numpy array of embeddings
        """
        if self.location_texts is None:
            raise ValueError("No location data loaded. Call load_and_process_data first.")
        
        # Generate embeddings
        embeddings = self.model.encode(self.location_texts, show_progress_bar=True)
        self.embedding_dim = embeddings.shape[1]
        
        return embeddings
    
    def build_faiss_index(self, embeddings: np.ndarray) -> faiss.IndexFlatL2:
        """
        Build a FAISS index for fast similarity search.
        
        Args:
            embeddings: Numpy array of embeddings
            
        Returns:
            FAISS index
        """
        # Create a FAISS index
        index = faiss.IndexFlatL2(self.embedding_dim)
        index.add(embeddings.astype('float32'))
        self.index = index
        
        return index
    
    def save_resources(self, save_dir: str = "./data") -> None:
        """
        Save the embeddings, index, and processed data.
        
        Args:
            save_dir: Directory to save resources
        """
        os.makedirs(save_dir, exist_ok=True)
        
        # Save the DataFrame
        self.locations_df.to_pickle(os.path.join(save_dir, "locations_df.pkl"))
        
        # Save the FAISS index
        faiss.write_index(self.index, os.path.join(save_dir, "locations.index"))
        
        # Save the model name for future reference
        with open(os.path.join(save_dir, "model_info.pkl"), "wb") as f:
            pickle.dump({"model_name": self.model.get_sentence_embedding_dimension()}, f)
    
    def load_resources(self, save_dir: str = "./data") -> Tuple[pd.DataFrame, faiss.IndexFlatL2]:
        """
        Load saved embeddings, index, and processed data.
        
        Args:
            save_dir: Directory where resources are saved
            
        Returns:
            Tuple of (DataFrame, FAISS index)
        """
        # Load the DataFrame
        self.locations_df = pd.read_pickle(os.path.join(save_dir, "locations_df.pkl"))
        self.location_texts = self.locations_df['combined_text'].tolist()
        
        # Load the FAISS index
        self.index = faiss.read_index(os.path.join(save_dir, "locations.index"))
        self.embedding_dim = self.index.d
        
        return self.locations_df, self.index
    
    def embed_query(self, query: str) -> np.ndarray:
        """
        Embed a user query.
        
        Args:
            query: User query string
            
        Returns:
            Numpy array of query embedding
        """
        return self.model.encode([query])[0].reshape(1, -1).astype('float32')
    
    def retrieve_similar_locations(self, query_embedding: np.ndarray, k: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve the k most similar locations to the query.
        
        Args:
            query_embedding: Embedded query
            k: Number of similar locations to retrieve (max 10)
            
        Returns:
            List of dictionaries containing location information
        """
        if self.index is None:
            raise ValueError("No index available. Either create one with build_faiss_index or load one with load_resources.")
        
        # Limit k to maximum 10 results
        k = min(k, 10)
        
        # Search the index
        distances, indices = self.index.search(query_embedding, k)
        
        # Get the corresponding locations
        similar_locations = []
        for i, idx in enumerate(indices[0]):
            location = self.locations_df.iloc[idx].to_dict()
            location['similarity_score'] = float(distances[0][i])
            similar_locations.append(location)
        
        return similar_locations


def initialize_embedder(csv_path: str, force_rebuild: bool = False) -> LocationEmbedder:
    """
    Initialize the LocationEmbedder, either by loading saved resources or creating new ones.
    
    Args:
        csv_path: Path to the locations.csv file
        force_rebuild: Whether to force rebuilding the embeddings and index
        
    Returns:
        Initialized LocationEmbedder
    """
    embedder = LocationEmbedder()
    data_dir = "./data"
    
    # Check if saved resources exist and we're not forcing a rebuild
    if os.path.exists(data_dir) and not force_rebuild:
        try:
            print("Loading saved embeddings and index...")
            embedder.load_resources(data_dir)
            print("Resources loaded successfully!")
            return embedder
        except Exception as e:
            print(f"Error loading resources: {e}")
            print("Building new embeddings and index...")
    
    # Create new embeddings and index
    print("Processing location data...")
    embedder.load_and_process_data(csv_path)
    
    print("Creating embeddings...")
    embeddings = embedder.create_embeddings()
    
    print("Building FAISS index...")
    embedder.build_faiss_index(embeddings)
    
    print("Saving resources...")
    embedder.save_resources(data_dir)
    
    return embedder
