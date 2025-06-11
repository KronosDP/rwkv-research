import os
import random

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# --- Language Definitions ---

def is_in_L1(s):
    """Checks if a string is in L1 = {(ab)^n | n >= 0}."""
    if s == "":
        return True
    if len(s) % 2 != 0:
        return False
    for i in range(0, len(s), 2):
        if s[i:i+2] != 'ab':
            return False
    return True

def is_in_L2(s):
    """Checks if a string is in L2 = {w in {a,b}* | #a(w) is even}."""
    return s.count('a') % 2 == 0

def is_in_L3(s):
    """Checks if a string is in L3 = {w in {a,b,c}* | 'abbccc' is a substring}."""
    return 'abbccc' in s

def is_in_L4(s):
    """Checks if a string is in L4 = L1 U L3."""
    return is_in_L1(s) or is_in_L3(s)

# --- Near-Miss Generation ---

def generate_near_miss(s, language_check_func, alphabet):
    """
    Generates a 'near-miss' string with an edit distance of 1, 2, or 3
    that is guaranteed not to be in the language.
    """
    for _ in range(50): # Try 50 times to find a valid near-miss
        temp_s = list(s)
        edit_distance = random.randint(1, 3)

        for _ in range(edit_distance):
            action = random.choice(['insert', 'delete', 'substitute'])
            
            if action == 'insert' and len(temp_s) < 300: # Max length constraint
                pos = random.randint(0, len(temp_s))
                char = random.choice(alphabet)
                temp_s.insert(pos, char)

            elif action == 'delete' and len(temp_s) > 1: # Min length constraint
                pos = random.randint(0, len(temp_s) - 1)
                temp_s.pop(pos)

            elif action == 'substitute' and len(temp_s) > 0:
                pos = random.randint(0, len(temp_s) - 1)
                original_char = temp_s[pos]
                possible_substitutions = [c for c in alphabet if c != original_char]
                if possible_substitutions:
                    temp_s[pos] = random.choice(possible_substitutions)
        
        near_miss_str = "".join(temp_s)
        if not language_check_func(near_miss_str):
            return near_miss_str
            
    return None # Return None if a valid near-miss can't be generated

# --- Dataset Generation Core ---

def generate_language_data(lang_name, check_func, alphabet, num_samples, min_len, max_len):
    """Generates a dataset for a given regular language."""
    data = {'string': [], 'label': [], 'type': []}
    
    # 1. Generate Positive Samples
    positive_count = 0
    while positive_count < num_samples // 2:
        if lang_name == 'L1':
            n = random.randint(min_len // 2, max_len // 2)
            s = 'ab' * n
        elif lang_name == 'L2':
            s_len = random.randint(min_len, max_len)
            s = "".join(random.choices(alphabet, k=s_len))
            if is_in_L2(s):
                pass
            else: # Flip one char to make count even
                if 'b' in s:
                    s = s.replace('b', 'a', 1)
                else: # all 'a's
                    s = s[:-1] # remove one 'a'
        elif lang_name == 'L3':
            s_len = random.randint(min_len, max_len)
            pre_len = random.randint(0, s_len - 6)
            post_len = s_len - 6 - pre_len
            pre = "".join(random.choices(alphabet, k=pre_len))
            post = "".join(random.choices(alphabet, k=post_len))
            s = pre + 'abbccc' + post
        elif lang_name == 'L4':
            if random.random() > 0.5: # Generate from L1
                 n = random.randint(min_len // 2, max_len // 2)
                 s = 'ab' * n
            else: # Generate from L3
                s_len = random.randint(min_len, max_len)
                pre_len = random.randint(0, s_len - 6)
                post_len = s_len - 6 - pre_len
                pre = "".join(random.choices(alphabet, k=pre_len))
                post = "".join(random.choices(alphabet, k=post_len))
                s = pre + 'abbccc' + post
        
        if min_len <= len(s) <= max_len and check_func(s):
            data['string'].append(s)
            data['label'].append(1)
            data['type'].append('positive')
            positive_count += 1
            
    # 2. Generate Negative and Near-Miss Samples
    negative_count = 0
    near_miss_count = 0
    
    positive_samples_for_nm = [s for s, t in zip(data['string'], data['type']) if t == 'positive']

    while negative_count + near_miss_count < num_samples // 2:
        # Generate near-misses for half of the negative samples
        if random.random() > 0.5 and near_miss_count < num_samples // 4 and positive_samples_for_nm:
            base_s = random.choice(positive_samples_for_nm)
            s = generate_near_miss(base_s, check_func, alphabet)
            if s and min_len <= len(s) <= max_len:
                data['string'].append(s)
                data['label'].append(0)
                data['type'].append('near-miss')
                near_miss_count += 1
        # Generate random negative samples
        else:
            s_len = random.randint(min_len, max_len)
            s = "".join(random.choices(alphabet, k=s_len))
            if not check_func(s):
                data['string'].append(s)
                data['label'].append(0)
                data['type'].append('negative')
                negative_count += 1
                
    return pd.DataFrame(data)

# --- Main Execution ---

def main():
    """Main function to generate all datasets and reports."""
    
    languages = {
        'L1': {'func': is_in_L1, 'alphabet': 'ab'},
        'L2': {'func': is_in_L2, 'alphabet': 'ab'},
        'L3': {'func': is_in_L3, 'alphabet': 'abc'},
        'L4': {'func': is_in_L4, 'alphabet': 'abc'}
    }
    
    # Sequence length configurations for training and testing
    configs = {
        'train_50_test_100_200': {'train_len': (40, 50), 'val_len': (90, 100), 'test_len': (190, 200)},
        'train_100_test_200_300': {'train_len': (90, 100), 'val_len': (190, 200), 'test_len': (290, 300)}
    }
    
    total_samples_per_lang = 1_000_000 # Total samples for each language dataset
    
    # Create base directory for datasets
    if not os.path.exists('datasets'):
        os.makedirs('datasets')
        
    full_report = ""

    for lang_name, props in languages.items():
        lang_dir = os.path.join('datasets', lang_name)
        if not os.path.exists(lang_dir):
            os.makedirs(lang_dir)
            
        lang_report = f"--- Dataset Generation Report for {lang_name} ---\n\n"
        print(f"Generating datasets for {lang_name}...")
        
        for config_name, len_conf in configs.items():
            config_dir = os.path.join(lang_dir, config_name)
            if not os.path.exists(config_dir):
                os.makedirs(config_dir)

            lang_report += f"Configuration: {config_name}\n"

            # Generate combined dataset first
            all_data = generate_language_data(
                lang_name, props['func'], props['alphabet'], 
                total_samples_per_lang, 
                min(len_conf['train_len'][0], len_conf['val_len'][0], len_conf['test_len'][0]),
                max(len_conf['train_len'][1], len_conf['val_len'][1], len_conf['test_len'][1])
            )

            # Filter data for each set based on length
            train_data = all_data[all_data['string'].str.len().between(*len_conf['train_len'])]
            val_data = all_data[all_data['string'].str.len().between(*len_conf['val_len'])]
            test_data = all_data[all_data['string'].str.len().between(*len_conf['test_len'])]

            # Shuffle the datasets
            train_data = train_data.sample(frac=1, random_state=42).reset_index(drop=True)
            val_data = val_data.sample(frac=1, random_state=42).reset_index(drop=True)
            test_data = test_data.sample(frac=1, random_state=42).reset_index(drop=True)

            # Save to CSV
            train_data.to_csv(os.path.join(config_dir, 'train.csv'), index=False)
            val_data.to_csv(os.path.join(config_dir, 'validation.csv'), index=False)
            test_data.to_csv(os.path.join(config_dir, 'test.csv'), index=False)
            
            # Report generation
            lang_report += f"  Training Set ({len_conf['train_len'][0]}-{len_conf['train_len'][1]} tokens):\n"
            lang_report += f"    Total samples: {len(train_data)}\n"
            lang_report += f"{train_data['type'].value_counts().to_string()}\n"
            
            lang_report += f"  Validation Set ({len_conf['val_len'][0]}-{len_conf['val_len'][1]} tokens):\n"
            lang_report += f"    Total samples: {len(val_data)}\n"
            lang_report += f"{val_data['type'].value_counts().to_string()}\n"
            
            lang_report += f"  Test Set ({len_conf['test_len'][0]}-{len_conf['test_len'][1]} tokens):\n"
            lang_report += f"    Total samples: {len(test_data)}\n"
            lang_report += f"{test_data['type'].value_counts().to_string()}\n\n"

        full_report += lang_report + "\n"

    # Save the full report to a file
    with open('datasets/generation_summary_report.txt', 'w') as f:
        f.write(full_report)
        
    print("\nAll datasets generated successfully!")
    print("A detailed report can be found in 'datasets/generation_summary_report.txt'")


if __name__ == '__main__':
    main()
