import os
import re

directory = "."

def process_file(filepath):
    with open(filepath, 'r') as f:
        content = f.read()
    
    if "width: 1304" not in content:
        return False

    # Check if already wrapped
    if "maxWidth: 528" in content and "margin: '0 auto'" in content and "<Header" in content:
        # print(f"Skipping {filepath}: Already wrapped")
        return False

    # Relaxed regex
    # matches <Header ... /> with any props
    header_pattern = re.compile(r'([ \t]*)(<Header\s+[^>]+/>)', re.DOTALL)
    
    match = header_pattern.search(content)
    if match:
        indent = match.group(1)
        header_tag = match.group(2)
        
        # Ensure we are replacing the main Header, not something else (though usually only one)
        
        wrapper_start = f'{indent}<div style={{{{width: \'100%\', maxWidth: 528, margin: \'0 auto\', display: \'flex\', flexDirection: \'column\'}}}}>'
        wrapper_end = f'{indent}</div>'
        
        # We need to handle if indent is empty (e.g. start of line)
        # But usually it's indented inside div.
        
        new_header_block = f'{wrapper_start}\n{indent}  {header_tag}\n{wrapper_end}'
        
        # Replace only the first occurrence
        new_content = content.replace(match.group(0), new_header_block, 1)
        
        if new_content != content:
            with open(filepath, 'w') as f:
                f.write(new_content)
            print(f"Updated {filepath}")
            return True
    else:
        print(f"Regex failed for {filepath}")
            
    return False

count = 0
files = os.listdir(directory)
files.sort()
for filename in files:
    if filename.endswith(".js") and filename.startswith("Literacy"):
        if process_file(filename):
            count += 1

print(f"Total files updated: {count}")
