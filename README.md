---
title: DataMeasurementsTool
emoji: 🤗
colorFrom: indigo
colorTo: red
sdk: streamlit
sdk_version: 1.0.0
app_file: app.py
pinned: false
---

# Data Measurements Tool

🚧 Doing Construction 🚧

[![Generic badge](https://img.shields.io/badge/🤗-Open%20In%20Spaces-blue.svg)](https://huggingface.co/spaces/huggingface/data-measurements-tool)

For more information, check out out [blog post](https://huggingface.co/blog/data-measurements-tool)!

# How to run:

After cloning, (and potentially setting up your virtual environment) run:

`pip install -r requirements.txt`

This installs all the requirements for the tool.

## Command Line Interface

From there, you can measure different aspects of different datasets by running `run_data_measurements.py` with different options.
The options specify the HF Dataset, the Dataset config, the Dataset columns being measured, the measurements to use, and further details about caching and saving.

To see the full list of options, do:

`python3 run_data_measurements.py -h` or `python3 run_data_measurements.py --help`

Example for hate_speech18 dataset:

         `python3 run_data_measurements.py --dataset="hate_speech18" --config="default" --split="train" --feature="text"`

Example for getting *just* the nPMI measurement from hate_speech18:

         `python3 run_data_measurements.py --dataset=hate_speech18 --config default --split train --feature text --calculation npmi`


Example for IMDB dataset:

` python3 run_data_measurements.py --dataset="imdb" --config="plain_text" --split="train" --label_field="label" --feature="text"`


## User Interface

`streamlit run app.py`

# Configuration

`title`: _string_  
Display title for the Space

`emoji`: _string_  
Space emoji (emoji-only character allowed)

`colorFrom`: _string_  
Color for Thumbnail gradient (red, yellow, green, blue, indigo, purple, pink, gray)

`colorTo`: _string_  
Color for Thumbnail gradient (red, yellow, green, blue, indigo, purple, pink, gray)

`sdk`: _string_  
Can be either `gradio` or `streamlit`

`sdk_version` : `1.0.0`  
Only applicable for `streamlit` SDK.  
See [doc](https://hf.co/docs/hub/spaces) for more info on supported versions.

`app_file`: _string_  
Path to your main application file (which contains either `gradio` or `streamlit` Python code).  
Path is relative to the root of the repository.

`pinned`: _boolean_  
Whether the Space stays on top of your list.
