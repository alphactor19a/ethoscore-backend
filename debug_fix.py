import re

filepath = "LiteracyRedfin.js"
with open(filepath, 'r') as f:
    content = f.read()

print(f"File length: {len(content)}")
if "width: 1304" in content:
    print("Found width: 1304")
else:
    print("NOT Found width: 1304")

header_pattern = re.compile(r'([ \t]*)(<Header\s+[^>]+/>)', re.DOTALL)
match = header_pattern.search(content)
if match:
    print(f"Match found: '{match.group(0)}'")
    print(f"Indent: '{match.group(1)}'")
    print(f"Tag: '{match.group(2)}'")
else:
    print("No regex match")
