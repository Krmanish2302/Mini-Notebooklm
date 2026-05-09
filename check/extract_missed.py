import re
import os

root_dir = r'c:\Users\kumar\OneDrive\Desktop\anti'
text_file = os.path.join(root_dir, 'text.txt')

with open(text_file, 'r', encoding='utf-8') as f:
    content = f.read()

# Manual extraction for specific blocks
def extract_block(header_pattern):
    match = re.search(header_pattern + r'.*?```(?:python|yaml|dockerfile|yaml|text)?\n(.*?)\n```', content, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1)
    return None

# 1. config.yaml
config_yaml = extract_block(r'\*\*Configuration Schema \(config.yaml\)\*\*:')
if config_yaml:
    with open(os.path.join(root_dir, 'config.yaml'), 'w', encoding='utf-8') as f:
        f.write(config_yaml)
    print("Saved config.yaml")

# 2. Source Cleaners
pdf_cleaner = extract_block(r'\*\*PDF Cleaner\*\*:')
if pdf_cleaner:
    path = os.path.join(root_dir, 'src/ingestion/preprocessing/source_cleaners/pdf_cleaner.py')
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(pdf_cleaner)
    print("Saved pdf_cleaner.py")

website_cleaner = extract_block(r'\*\*Website Cleaner\*\*:')
if website_cleaner:
    path = os.path.join(root_dir, 'src/ingestion/preprocessing/source_cleaners/website_cleaner.py')
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(website_cleaner)
    print("Saved website_cleaner.py")

youtube_cleaner = extract_block(r'\*\*YouTube Cleaner\*\*:')
if youtube_cleaner:
    path = os.path.join(root_dir, 'src/ingestion/preprocessing/source_cleaners/youtube_cleaner.py')
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(youtube_cleaner)
    print("Saved youtube_cleaner.py")

# 3. tests/test_ingestion.py
test_ingestion = extract_block(r'# tests/test_ingestion.py')
if test_ingestion:
    path = os.path.join(root_dir, 'tests/test_ingestion.py')
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(test_ingestion)
    print("Saved test_ingestion.py")

# 4. verify_pipeline.py
verify_pipeline = extract_block(r'# Verification script for each component')
if verify_pipeline:
    path = os.path.join(root_dir, 'src/verify_pipeline.py')
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(verify_pipeline)
    print("Saved verify_pipeline.py")
