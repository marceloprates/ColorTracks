# ColorTracks

This project contains a Streamlit application that can be run locally.

## Prerequisites

- Python 3.11
- pip (Python package installer)

## Setup and Installation

1. Install Python 3.11 if not already installed:
    - For Ubuntu/Debian:
      ```bash
      sudo apt update
      sudo apt install python3.11 python3.11-venv
      ```
    - For macOS (using Homebrew):
      ```bash
      brew install python@3.11
      ```
    - For Windows:
      Download the installer from [python.org](https://www.python.org/downloads/release/python-3110/)

2. Create and activate a virtual environment (recommended):
    ```bash
    python3.11 -m venv venv
    # On Windows
    # venv\Scripts\activate
    # On Unix or MacOS
    source venv/bin/activate
    ```

3. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

## Running the Application

Start the Streamlit application with:

```bash
streamlit run src/app.py
```

The application should open in your default web browser. If it doesn't, navigate to the URL shown in the terminal (typically http://localhost:8501).

## Project Structure

```
Pegeout/
├── src/
│   └── app.py        # Main Streamlit application
├── src/
│   ├── app.py        # Main Streamlit application
│   └── src.py        # Source code for application logic
├── requirements.txt  # Project dependencies
└── README.md         # This file
```

## License

[Specify your license here]