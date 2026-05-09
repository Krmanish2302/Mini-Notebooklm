import re
import os

root_dir = r'c:\Users\kumar\OneDrive\Desktop\anti'
text_file = os.path.join(root_dir, 'text.txt')

with open(text_file, 'r', encoding='utf-8') as f:
    content = f.read()

def extract_block(header_pattern):
    # Match the header and the following code block
    match = re.search(header_pattern + r'.*?```(?:python|yaml|dockerfile|yaml|text|json)?\n(.*?)\n```', content, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1)
    return None

# Verification script
verify_pipeline = extract_block(r'After implementing EACH phase, verify:')
if verify_pipeline:
    path = os.path.join(root_dir, 'src/verify_pipeline.py')
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(verify_pipeline)
    print("Saved verify_pipeline.py")

# JSON Format example
pipeline_format = extract_block(r'All pipelines must return consistent format:')
if pipeline_format:
    path = os.path.join(root_dir, 'docs/PIPELINE_FORMAT.json')
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(pipeline_format)
    print("Saved PIPELINE_FORMAT.json")
