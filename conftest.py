import os
import sys

# Make sure the repository root is importable so that the packages shipped in
# this repository (e.g. ``topo_data_util``) can be imported during testing,
# regardless of the directory pytest is invoked from.
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
