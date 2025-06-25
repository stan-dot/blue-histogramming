import re
import json

with open("events-to-emit-11.json") as f:
    text = f.read()

pattern = r"SENDING b'(.*?)' to"
matches = re.findall(pattern, text)

json_docs = []
for m in matches:
    json_str = m.encode("utf-8").decode("unicode_escape")
    try:
        json_docs.append(json.loads(json_str))
    except Exception as e:
        print("Failed to parse:", json_str, e)

# Write all extracted JSON objects to a new file
with open("extracted_events.json", "w") as out_f:
    json.dump(json_docs, out_f, indent=2)
