#!/bin/bash
# Convenience script to run the vision location detector application

# Activate virtual environment
source venv/bin/activate

# Run the application
python -m src.main "$@"
