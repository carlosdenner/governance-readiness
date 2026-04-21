import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import warnings

# [debug]
def inspect_columns(df):
    print("Columns available:", df.columns.tolist())
    print("Sample of 'Known AI Technology':")
    print(df['Known AI Technology'].dropna().head(10))
    print("Sample of 'Potential AI Technology':")
    if 'Potential AI Technology' in df.columns:
        print(df['Potential AI Technology'].dropna().head(10))
    print("Sample of 'Known AI Goal':")
    if 'Known AI Goal' in df.columns:
        print(df['Known AI Goal'].dropna().head(10))

def experiment():
    warnings.simplefilter(action='ignore', category=FutureWarning)
    
    # 1. Load Dataset
    file_path = '../astalabs_discovery_all_data.csv'
    try:
        df = pd.read_csv(file_path, low_memory=False)
    except FileNotFoundError:
        print("File not found at ../, trying current directory...")
        try:
            df = pd.read_csv('astalabs_discovery_all_data.csv', low_memory=False)
        except FileNotFoundError:
            print("Dataset not found.")
            return

    # Filter for AIID incidents
    df_aiid = df[df['source_table'] == 'aiid_incidents'].copy()
    
    # DEBUG: Inspect columns to find best text fields
    # inspect_columns(df_aiid)

    # 2. Extract Year
    df_aiid['date'] = pd.to_datetime(df_aiid['date'], errors='coerce')
    df_aiid = df_aiid.dropna(subset=['date'])
    df_aiid['year'] = df_aiid['date'].dt.year.astype(int)
    
    # Filter relevant years for trend analysis (2014-2024)
    df_aiid = df_aiid[(df_aiid['year'] >= 2014) & (df_aiid['year'] <= 2024)]

    # 3. Classify Technology
    # Combine relevant columns to improve recall
    text_cols = ['Known AI Technology', 'Potential AI Technology', 'Known AI Goal', 'title', 'description', 'summary']
    
    # Create a consolidated text field for search
    df_aiid['combined_text'] = ""
    for col in text_cols:
        if col in df_aiid.columns:
            df_aiid['combined_text'] += df_aiid[col].fillna('').astype(str) + " "
            
    df_aiid['combined_text'] = df_aiid['combined_text'].str.lower()

    def classify_tech(text):
        # Keywords for Generative / Content AI
        gen_keywords = [
            'generative', 'llm', 'gpt', 'chat', 'diffusion', 'dall-e', 'midjourney', 
            'stable diffusion', 'text-to-image', 'chatbot', 'language model', 
            'bert', 'transformer', 'deepfake', 'hallucination', 'gemini', 'copilot', 'llama',
            'content generation', 'stylegan', 'prompt'
        ]
        
        # Keywords for Robotic / Physical AI
        bot_keywords = [
            'robot', 'autonomous', 'self-driving', 'drone', 'uav', 'tesla', 'waymo', 
            'cruise', 'vehicle', 'autopilot', 'driverless', 'car', 'physical', 
            'manufacturing', 'industrial', 'robotic', 'humanoid'
        ]
        
        is_gen = any(k in text for k in gen_keywords)
        is_bot = any(k in text for k in bot_keywords)
        
        if is_gen and not is_bot:
            return 'Generative AI'
        elif is_bot and not is_gen:
            return 'Robotic/Autonomous'
        elif is_gen and is_bot:
            return 'Mixed'
        else:
            return 'Other'

    df_aiid['tech_category'] = df_aiid['combined_text'].apply(classify_tech)

    # 4. Aggregate Counts
    pivot = df_aiid.groupby(['year', 'tech_category']).size().unstack(fill_value=0)
    
    # Ensure key columns exist
    for col in ['Generative AI', 'Robotic/Autonomous', 'Other']:
        if col not in pivot.columns:
            pivot[col] = 0
            
    target_df = pivot[['Generative AI', 'Robotic/Autonomous']].copy()
    
    # Calculate Growth Rates and Share
    growth = target_df.pct_change() * 100
    total_classified = target_df['Generative AI'] + target_df['Robotic/Autonomous']
    
    # Handle division by zero for share calculation
    gen_share = pd.Series(0.0, index=target_df.index)
    mask = total_classified > 0
    gen_share[mask] = (target_df.loc[mask, 'Generative AI'] / total_classified[mask]) * 100

    # 5. Output Deliverables
    print("--- Time-series: Incident Counts by Technology Type ---")
    print(target_df)
    
    print("\n--- Year-over-Year Growth Rates (%) ---")
    print(growth.round(1).replace({np.inf: 'Inf', np.nan: '-'}))
    
    print("\n--- Generative AI Share of Total Classified Incidents (%) ---")
    print(gen_share.round(1))

    # Check Harm Domain Association (Hypothesis Validation)
    print("\n--- Harm Domain Distribution (Top 3) by Tech Category ---")
    if 'Harm Domain' in df_aiid.columns:
        subset = df_aiid[df_aiid['tech_category'].isin(['Generative AI', 'Robotic/Autonomous'])].copy()
        
        # Robustly handle Harm Domain cleaning
        # Fill NaNs with empty string, force to string, then split
        subset['Harm_Primary'] = subset['Harm Domain'].fillna('').astype(str).apply(lambda x: x.split(',')[0].strip() if x else 'Unknown')
        
        # Remove 'Unknown' or empty if desired, or keep to show missing data
        subset = subset[subset['Harm_Primary'] != 'Unknown']
        subset = subset[subset['Harm_Primary'] != 'nan']
        
        if not subset.empty:
            ct = pd.crosstab(subset['tech_category'], subset['Harm_Primary'])
            # Normalize row-wise percentages
            ct_norm = ct.div(ct.sum(axis=1), axis=0) * 100
            
            # Show top 3 columns for each row
            for cat in ct_norm.index:
                print(f"\n{cat}:")
                print(ct_norm.loc[cat].sort_values(ascending=False).head(3).round(1))
        else:
            print("No valid Harm Domain data found for classified incidents.")

    # 6. Plotting
    plt.figure(figsize=(12, 6))
    plt.plot(target_df.index, target_df['Generative AI'], marker='o', linewidth=2.5, label='Generative AI')
    plt.plot(target_df.index, target_df['Robotic/Autonomous'], marker='s', linewidth=2.5, label='Robotic/Autonomous')
    
    plt.title('The Generative AI Explosion: Incident Trends (2014-2024)')
    plt.xlabel('Year')
    plt.ylabel('Number of Recorded Incidents')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    experiment()