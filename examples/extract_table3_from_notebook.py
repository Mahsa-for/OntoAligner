import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
import json
import os

# Load the notebook
notebook_path = "examples/OntoAlignerPipeline-Exp.ipynb"
with open(notebook_path) as f:
    nb = nbformat.read(f, as_version=4)

# Execute the notebook
ep = ExecutePreprocessor(timeout=600, kernel_name='python3')
ep.preprocess(nb, {'metadata': {'path': './examples'}})

# Extract Table 3 data from output cells
table3_data = []
for cell in nb.cells:
    if cell.cell_type == "code" and cell.outputs:
        for output in cell.outputs:
            if "text/plain" in output.get("data", {}):
                text = output["data"]["text/plain"]
                if "Precision" in text and "F1" in text:
                    # Parse lines that look like rows
                    lines = text.split("\n")
                    for line in lines:
                        if "(" in line and "%" not in line:
                            parts = line.strip().split()
                            try:
                                method = parts[0]
                                model = " ".join(parts[1:-6])
                                precision = float(parts[-6])
                                recall = float(parts[-5])
                                f1 = float(parts[-4])
                                time = float(parts[-3])
                                params = " ".join(parts[-2:])
                                table3_data.append({
                                    "Method": method,
                                    "Model (Encoder)": model,
                                    "Precision": precision,
                                    "Recall": recall,
                                    "F1": f1,
                                    "Time (s)": time,
                                    "Parameters": params
                                })
                            except:
                                pass

# Save to JSON
output_file = "ontoaligner_table3_from_notebook.json"
with open(output_file, "w") as f:
    json.dump(table3_data, f, indent=2)

print(f"Table 3 data extracted and saved to {output_file}")
