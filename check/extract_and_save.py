import os

# Project root
root_dir = r'c:\Users\kumar\OneDrive\Desktop\anti'

with open(os.path.join(root_dir, 'text.txt'), 'r', encoding='utf-8') as f:
    lines = f.readlines()

files = []
current_file = None
in_code_block = False
code_content = []

for line in lines:
    if line.startswith('**File**: ') or line.startswith('**File**:'):
        current_file = line.split('`')[1] if '`' in line else line.split(':')[1].strip()
    elif line.startswith('**Configuration Schema (config.yaml)**:'):
        current_file = 'config.yaml'
    
    if line.strip().startswith('```') and current_file:
        if not in_code_block:
            in_code_block = True
            code_content = []
        else:
            in_code_block = False
            files.append((current_file, ''.join(code_content)))
            current_file = None
    elif in_code_block:
        code_content.append(line)

print(f"Extracted {len(files)} files. Writing to disk...")

for file_path, content in files:
    full_path = os.path.join(root_dir, file_path)
    os.makedirs(os.path.dirname(full_path), exist_ok=True)
    with open(full_path, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"Created: {file_path}")

print("Extraction complete!")
