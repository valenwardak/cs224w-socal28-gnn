# =============================================================================
# File:        main_gcn.py
# Project:     CS224W Fall 2026, Project Type 1
#              GNN event detection on the SoCal 28-Bus dataset
# Authors:     Valen Wardak and Landon Choy, Stanford University
# Created:     2026-10-29, reduced to a wrapper 2026-11-07
#
# Purpose:     Train and score the GCN. The graph models now live in
#              main_gnn.py, whose --kind option defaults to gcn; this file
#              exists so that every earlier command and notebook cell that
#              calls main_gcn.py keeps working unchanged. It runs
#              main_gnn.main() with the arguments as given.
#
# Usage:
#   python main_gcn.py --run-name gcn_v1
#   Options: python main_gnn.py --help
#
# Functions:
#   none of its own; main comes from main_gnn
# =============================================================================

# Project: the graph model script's entry point
from main_gnn import main


# Run when called as a script
if __name__ == "__main__":
    # Go
    main()
