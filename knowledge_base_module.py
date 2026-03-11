import os
import json
from pathlib import Path

class KnowledgeBaseModule:
    def __init__(self, data_dir="data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        self.context_data = ""
        self.load_data()

    def load_data(self):
        print("Loading knowledge data...")
        all_text = []
        
        # Load all txt, md, and json files from the data directory
        for file_path in self.data_dir.glob("*"):
            if file_path.suffix in [".txt", ".md"]:
                with open(file_path, "r", encoding="utf-8") as f:
                    all_text.append(f"Source: {file_path.name}\n{f.read()}\n")
            elif file_path.suffix == ".json":
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        all_text.append(f"Source: {file_path.name}\n{json.dumps(data, indent=2)}\n")
                except Exception as e:
                    print(f"Error loading {file_path}: {e}")
        
        # Also check root for the existing hospital datasets if they are still there
        root_dir = Path(".")
        for json_file in root_dir.glob("hospital_reception_dataset*.json"):
             with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                all_text.append(f"Source: {json_file.name}\n{json.dumps(data[:50], indent=2)}\n") # Only first 50 entries to avoid context limit
        
        self.context_data = "\n".join(all_text)
        print(f"Knowledge data loaded. Length: {len(self.context_data)} characters.")

    def get_context(self):
        return self.context_data
