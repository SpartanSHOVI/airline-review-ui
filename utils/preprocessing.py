import pandas as pd
import numpy as np
import re

def extract_verification(text):
    if pd.isna(text):
        return 0, str(text)
    if '|' in str(text):
        parts = str(text).split('|', 1)
        status = parts[0].strip().lower()
        content = parts[1].strip()
        is_verified = 1 if 'verified' in status else 0
        return is_verified, content
    return 0, str(text)

def initial_cleaning(df):
    df_clean = df.copy()
    
    # Try to find review text column
    text_cols = [c for c in df_clean.columns if 'review' in c.lower() or 'text' in c.lower() or 'body' in c.lower()]
    if text_cols:
        text_col = text_cols[0]
        # Extract verification
        verified_status = []
        clean_text = []
        for val in df_clean[text_col]:
            v, c = extract_verification(val)
            verified_status.append(v)
            clean_text.append(c)
        
        df_clean['is_verified'] = verified_status
        df_clean[text_col] = clean_text
        
    # Drop columns with > 70% missing values
    threshold = 0.7 * len(df_clean)
    df_clean = df_clean.dropna(thresh=threshold, axis=1)
    
    # Fill numerical missing values with median
    num_cols = df_clean.select_dtypes(include=[np.number]).columns
    for c in num_cols:
        df_clean[c] = df_clean[c].fillna(df_clean[c].median())
        
    # Fill categorical missing values with mode
    cat_cols = df_clean.select_dtypes(include=['object']).columns
    for c in cat_cols:
        df_clean[c] = df_clean[c].fillna(df_clean[c].mode()[0] if not df_clean[c].mode().empty else 'Unknown')
        
    return df_clean

def process_nlp(df, text_column):
    df_nlp = df.copy()
    
    def clean_text(text):
        if pd.isna(text): return ""
        # Lowercase
        text = str(text).lower()
        # Remove punctuation and numbers
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        return text
        
    df_nlp[f"{text_column}_cleaned"] = df_nlp[text_column].apply(clean_text)
    
    # Simple token count
    df_nlp[f"{text_column}_word_count"] = df_nlp[f"{text_column}_cleaned"].apply(lambda x: len(x.split()))
    
    return df_nlp

def engineer_features(df):
    df_eng = df.copy()
    
    # Identify categorical columns with low cardinality
    cat_cols = df_eng.select_dtypes(include=['object']).columns
    for c in cat_cols:
        if df_eng[c].nunique() < 10 and 'text' not in c.lower() and 'review' not in c.lower():
            # One hot encode
            dummies = pd.get_dummies(df_eng[c], prefix=c, drop_first=True)
            df_eng = pd.concat([df_eng, dummies], axis=1)
            df_eng.drop(c, axis=1, inplace=True)
            
    return df_eng
