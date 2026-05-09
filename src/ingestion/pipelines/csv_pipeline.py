import pandas as pd
from typing import Dict, Any

class CSVPipeline:
    """Extracts structured data from CSV/Excel."""
    
    @staticmethod
    def process(file_path: str, source_id: str) -> Dict[str, Any]:
        df = pd.read_csv(file_path) if file_path.endswith('.csv') else pd.read_excel(file_path)
        
        description = f"""
Dataset: {file_path}
Shape: {df.shape[0]} rows x {df.shape[1]} columns
Columns: {', '.join(df.columns.tolist())}

Sample Data:
{df.head(5).to_markdown()}

Statistics:
{df.describe().to_markdown()}
"""
        
        return {
            "content": description,
            "metadata": {
                "shape": df.shape,
                "columns": df.columns.tolist(),
                "source_id": source_id
            },
            "modality": "table"
        }