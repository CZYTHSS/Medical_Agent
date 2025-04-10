import json
import os
import pandas as pd

# Define the base path
CACHE_DIR = "/Users/czy/Desktop/sidegig/Medical_Agent/src/medical_agent/cache"

def save_df_to_cache(df: pd.DataFrame, filename: str):
    """
    Save a DataFrame as a Parquet file to the cache directory.
    
    Args:
        df (pd.DataFrame): The DataFrame to save.
        filename (str): The name of the Parquet file (without extension).
    """
    os.makedirs(CACHE_DIR, exist_ok=True)
    full_path = os.path.join(CACHE_DIR, f"{filename}.parquet")
    df.to_parquet(full_path, index=False)
    print(f"✅ Saved to {full_path}")

def load_df_from_cache(filename: str) -> pd.DataFrame:
    """
    Load a DataFrame from a Parquet file in the cache directory.
    
    Args:
        filename (str): The name of the Parquet file (without extension).
        
    Returns:
        pd.DataFrame: The loaded DataFrame.
    """
    full_path = os.path.join(CACHE_DIR, f"{filename}.parquet")
    if not os.path.exists(full_path):
        raise FileNotFoundError(f"❌ Cache file not found: {full_path}")
    df = pd.read_parquet(full_path)
    print(f"✅ Loaded from {full_path}")
    return df


def call_qwen_vl_api(state):
    completion = state["qwen"].chat.completions.create(
        model="qwen-vl-max-latest",
        messages=state['messages']
    )

    text = completion.choices[0].message.content
    return text


def safe_json_load(text):
    try:
        return json.loads(text)
    except:
        try:
            if text.startswith("```json"):
                text = text.split("```json")[1]
            if text.startswith("```"):
                text = text.split("```")[1] 
            if text.endswith("```"):
                text = text.split("```")[0]
            return json.loads(text)
        except Exception as e:
            print(f"Load Json Dict Failed. Error: {e}")
            return None

