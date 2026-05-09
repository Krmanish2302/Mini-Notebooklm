import re
import os

root_dir = r'c:\Users\kumar\OneDrive\Desktop\anti'
text_file = os.path.join(root_dir, 'text.txt')

with open(text_file, 'r', encoding='utf-8') as f:
    content = f.read()

# Find all code blocks
blocks = re.findall(r'(.*?)\n```(.*?)\n(.*?)\n```', content, re.DOTALL)

print(f"Found {len(blocks)} code blocks.")

for i, (pre_content, lang, code) in enumerate(blocks):
    # Try to find a file path in the preceding 5 lines
    lines = pre_content.strip().split('\n')
    last_lines = lines[-5:] if len(lines) > 5 else lines
    
    file_path = None
    for line in reversed(last_lines):
        # Look for **File**: [path]
        match = re.search(r'\*\*File\*\*:\s*`?([\w\./-]+)`?', line)
        if match:
            file_path = match.group(1)
            break
        # Look for **Files**: followed by list
        if '**Files**:' in line:
            # The next few lines might contain paths. This is hard to parse genericly.
            pass
        # Look for # filename.py at the start of code
        code_lines = code.strip().split('\n')
        if code_lines and code_lines[0].startswith('# ') and code_lines[0].endswith('.py'):
            file_path = code_lines[0][2:].strip()
            break

    if file_path:
        print(f"Block {i}: Found path {file_path}")
        full_path = os.path.join(root_dir, file_path)
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        with open(full_path, 'w', encoding='utf-8') as f:
            f.write(code)
    else:
        # Check for the special case of **Files**:
        # We know line 917 has **Files**:
        # Let's see if this block is one of those.
        # Actually, I'll just manually check blocks that have no file_path.
        print(f"Block {i}: NO PATH FOUND. Preceding: {last_lines[-1] if last_lines else 'None'}")
        # Print first line of code to identify it
        first_line = code.strip().split('\n')[0]
        print(f"  Code starts with: {first_line}")
