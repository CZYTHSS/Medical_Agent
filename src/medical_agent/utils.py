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


import tkinter as tk
from tkinter import ttk
import pandas as pd

# Fixed coronary segments
segments = [
    "左主干近段 (pLM)", "左主干中段 (mLM)", "左主干远段 (dLM)", 
    "右前降支近段 (pLAD)", "右前降支中段 (mLAD)", "右前降支远段 (dLAD)", 
    "左回旋支近段 (pLCX)", "左回旋支中段 (mLCX)", "左回旋支远段 (dLCX)",
    "右冠近段 (pRCA)", "右冠中段 (mRCA)", "右冠远段 (dRCA)",
    "第一对角支 (D1)", "第二对角支 (D2)", "中间支 (RI)",
    "第一钝缘支 (OM1)", "第二钝缘支 (OM2)", 
    "左室侧后降支 (L-PDA)", "右室侧降支 (L-PDA)",
    "左室侧后支 (L-PLB)", "右室侧后支 (R-PLB)"
]

# Table columns
columns = ["斑块种类", "类型", "症状", "大小(mm)", "狭窄程度", "闭塞"]

# Dropdown values for certain fields
dropdown_options = {
    "斑块种类": ["NONE", "软斑块（非钙化性斑块）", "混合密度斑块", "硬斑块（钙化性斑块）"],
    "狭窄程度": ["NONE", "局限性狭窄", "阶段性狭窄", "弥漫性狭窄"],
    "闭塞": ["NONE", "是", "否"]
}

def show_popup_with_df(df: pd.DataFrame):
    root = tk.Tk()
    root.title("冠脉结构化填写")

    # Header
    header_frame = ttk.Frame(root)
    header_frame.pack(padx=10, pady=5, fill="x")

    ttk.Label(header_frame, text="冠状动脉起源、走形及终止:").grid(row=0, column=0, sticky="w")
    ttk.Combobox(header_frame, values=["正常", "异常"], width=10).grid(row=0, column=1, padx=5)

    ttk.Label(header_frame, text="冠脉优势型:").grid(row=0, column=2, sticky="w")
    ttk.Combobox(header_frame, values=["右冠优势型", "左冠优势型", "均衡性"], width=12).grid(row=0, column=3, padx=5)

    # Table frame
    table_frame = ttk.Frame(root)
    table_frame.pack(padx=10, pady=5)

    # Table headers
    ttk.Label(table_frame, text="冠脉节段", borderwidth=1, relief="solid", width=20).grid(row=0, column=0)
    for j, col in enumerate(columns):
        ttk.Label(table_frame, text=col, borderwidth=1, relief="solid", width=15).grid(row=0, column=j+1)

    # Table rows
    for i, seg in enumerate(segments):
        ttk.Label(table_frame, text=seg, borderwidth=1, relief="solid", width=20).grid(row=i+1, column=0)
        for j, col in enumerate(columns):
            value = str(df.at[i, col]) if pd.notna(df.at[i, col]) else "NONE"

            if col in dropdown_options:
                options = dropdown_options[col]
                if value not in options:
                    options = ["NONE"] + options  # Add fallback
                var = tk.StringVar()
                cb = ttk.Combobox(table_frame, textvariable=var, values=options, width=13)
                cb.grid(row=i+1, column=j+1)
                cb.set(value)  # ✅ Set visible value
            else:
                var = tk.StringVar()
                entry = ttk.Entry(table_frame, textvariable=var, width=15)
                entry.grid(row=i+1, column=j+1)
                entry.insert(0, value)  # ✅ Set visible value

    root.mainloop()

# 🔧 Test with dummy DataFrame
def test_gui_with_dummy_df():
    data = {
        "冠脉节段": segments
    }
    for col in columns:
        data[col] = ["NONE"] * len(segments)
    df = pd.DataFrame(data)
    show_popup_with_df(df)


if __name__ == "__main__":
    # 🚀 Run the test
    test_gui_with_dummy_df()
