# loads raw text documents from disk

import os


def load_text_file(file_path: str) -> str:
    # read a text file and return its contents
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    print(f"Loaded file: {file_path} ({len(content):,} characters)")
    return content
