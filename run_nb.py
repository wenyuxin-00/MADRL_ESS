import traceback
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor

notebook_filename = "notebooks/madrl/train_madrl_grid.ipynb"

try:
    with open(notebook_filename, encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)
    ep = ExecutePreprocessor(timeout=600, kernel_name='python3')
    ep.preprocess(nb, {'metadata': {'path': 'notebooks/madrl/'}})
    print("Execution successful!")
except Exception as e:
    print("Execution failed!")
    traceback.print_exc()
