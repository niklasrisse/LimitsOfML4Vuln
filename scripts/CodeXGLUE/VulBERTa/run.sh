#!/bin/bash

# Check that the script is called with an argument
if [ $# -eq 0 ]; then
  echo "Error: No transformation provided. Usage: $0 <string>"
  exit 1
fi

# Assign the argument to a variable
transformation=$1

python $(dirname "$0")/run.py --trafo $1