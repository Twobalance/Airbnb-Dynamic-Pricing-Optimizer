import json
import re

def convert_py_to_ipynb(py_file, ipynb_file):
    with open(py_file, 'r') as f:
        content = f.read()

    # Split by # %% markers
    # The first block might be empty if file starts with # %%
    cells = []
    
    # We use regex to split but keep the type
    # actually simplistic split works if we assume consistent formatting
    parts = re.split(r'\n# %%', content)
    
    for part in parts:
        if not part.strip():
            continue
            
        part = part.strip()
        cell_type = "code"
        source_lines = part.splitlines(keepends=True)
        
        # Check if it starts with [markdown]
        if source_lines[0].strip().startswith('[markdown]'):
            cell_type = "markdown"
            # remove the [markdown] line
            source_lines = source_lines[1:]
        
        # Remove leading/trailing newlines from source but keep indentation?
        # Jupyter source is a list of strings
        # We should treat lines properly
        
        # Filter comments in markdown cells if they start with # (but markdown headers start with # too)
        # In the py file, markdown is often commented out? 
        # In my generated file, I used:
        # # %% [markdown]
        # # # Header
        # # text
        
        # So I need to uncomment lines in markdown cells
        if cell_type == "markdown":
            cleaned_lines = []
            for line in source_lines:
                if line.strip().startswith('# '):
                    cleaned_lines.append(line.replace('# ', '', 1))
                elif line.strip() == '#':
                    cleaned_lines.append('\n')
                else:
                    cleaned_lines.append(line)
            source_lines = cleaned_lines
        
        # For code cells, just keep as is
        
        cell = {
            "cell_type": cell_type,
            "metadata": {},
            "source": source_lines
        }
        
        if cell_type == "code":
            cell["execution_count"] = None
            cell["outputs"] = []
            
        cells.append(cell)

    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {
                    "name": "ipython",
                    "version": 3
                },
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.8.5"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }

    with open(ipynb_file, 'w') as f:
        json.dump(notebook, f, indent=1)
    
    print(f"Created {ipynb_file}")

if __name__ == "__main__":
    convert_py_to_ipynb('notebooks/notebook_content.py', 'Project_1_Dynamic_Pricing.ipynb')
