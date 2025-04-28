import json
import os
from langchain_text_splitters import MarkdownHeaderTextSplitter

def table_to_markdown(table):
    if not table:
        return ""
    markdown_lines = []
    header_row = table[0]
    cleaned_header = [str(cell) if cell is not None else "" for cell in header_row]
    markdown_lines.append("| " + " | ".join(cleaned_header) + " |")
    markdown_lines.append("| " + " | ".join(["---"] * len(cleaned_header)) + " |")
    for row in table[1:]:
        cleaned_row = [str(cell) if cell is not None else "" for cell in row]
        markdown_lines.append("| " + " | ".join(cleaned_row) + " |")
    return "\n".join(markdown_lines)

def qa_to_markdown(qa):
    if not qa:
        return ""
    question = qa.get("question", "")
    answer = qa.get("answer", "")
    explanation = qa.get("explanation", "")
    steps = qa.get("steps", [])
    markdown = ["### Question-Answer", f"- **Question**: {question}", f"- **Answer**: {answer}"]
    if explanation:
        markdown.append(f"- **Explanation**: {explanation}")
    if steps:
        markdown.append("- **Calculation Steps**:")
        for step in steps:
            op = step.get("op", "")
            arg1 = step.get("arg1", "")
            arg2 = step.get("arg2", "")
            res = step.get("res", "")
            markdown.append(f"  - {op}: {arg1}, {arg2} → {res}")
    return "\n".join(markdown)

def process_json_to_markdown(input_file, output_file):
    try:
        with open(input_file, "r", encoding="utf-8") as f:
            json_data = json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"{input_file} not found.")
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse {input_file}: {e}")
    
    markdown_content = ["# Financial Data from train.json", ""]
    for entry in json_data:
        entry_id = entry.get("id", "Unknown ID")
        filename = entry.get("filename", "Unknown File")
        markdown_content.append(f"# Entry: {entry_id} ({filename})")
        markdown_content.append("")
        pre_text = entry.get("pre_text", [])
        if pre_text:
            markdown_content.append("## Pre-Text")
            markdown_content.append(" ".join(pre_text))
            markdown_content.append("")
        table = entry.get("table", [])
        if table:
            markdown_content.append("## Table")
            markdown_content.append(table_to_markdown(table))
            markdown_content.append("")
        post_text = entry.get("post_text", [])
        if post_text:
            markdown_content.append("## Post-Text")
            markdown_content.append(" ".join(post_text))
            markdown_content.append("")
        qa = entry.get("qa", {})
        if qa:
            markdown_content.append(qa_to_markdown(qa))
            markdown_content.append("")
        annotation = entry.get("annotation", {})
        if annotation:
            markdown_content.append("## Annotation")
            for key, value in annotation.items():
                markdown_content.append(f"- **{key}**: {value}")
            markdown_content.append("")
        markdown_content.append("---")
        markdown_content.append("")
    
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("\n".join(markdown_content))
    return output_file

def load_markdown_document(file_path):
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        raise FileNotFoundError(f"{file_path} not found.")
    except Exception as e:
        raise ValueError(f"Error reading {file_path}: {e}")

def get_markdown_splits(markdown_content):
    headers_to_split_on = [("#", "Header 1"), ("##", "Header 2"), ("###", "Header 3")]
    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on, strip_headers=False)
    return markdown_splitter.split_text(markdown_content)