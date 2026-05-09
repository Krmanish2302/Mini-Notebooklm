import os
for root, dirs, files in os.walk('src'):
    init_path = os.path.join(root, '__init__.py')
    if not os.path.exists(init_path):
        with open(init_path, 'a') as f:
            pass
print("Created __init__.py in all subdirectories")
