#!/usr/bin/env bash

conda create --name LimitsOfMl4Vuln python=3.9.16
eval "$(conda shell.bash hook)"
conda activate LimitsOfMl4Vuln

pip install -r ./requirements.txt