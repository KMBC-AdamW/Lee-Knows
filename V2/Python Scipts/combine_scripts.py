import os
import json

# Folder containing individual JSON files
INPUT_FOLDER = "output"
OUTPUT_FILE = "output/combined_all.json"

combined_data = []

# Automatically include all .json files except the output file itself
for filename in os.listdir(INPUT_FOLDER):
    if filename.endswith(".json") and filename != os.path.basename(OUTPUT_FILE):
        file_path = os.path.join(INPUT_FOLDER, filename)
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                combined_data.extend(data)
                print(f"✅ Added {filename} with {len(data)} records.")
        except Exception as e:
            print(f"❌ Failed to read {filename}: {e}")

# Save combined file
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(combined_data, f, ensure_ascii=False, indent=2)

print(f"\n🎉 Combined file saved as {OUTPUT_FILE} with {len(combined_data)} total records.")