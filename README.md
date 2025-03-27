# Product Research and Sales Guide CrewAI Project

This project implements a CrewAI-based system with multiple AI agents working together to research products and create comprehensive sales guides.

## Overview

The system consists of four specialized agents:

1. **Searcher Agent**: Finds relevant product information online using web search APIs
2. **Scraper Agent**: Extracts detailed information from websites using web scraping techniques
3. **Analyst Agent**: Analyzes data for inconsistencies, errors, and gaps
4. **Writer Agent**: Creates comprehensive and persuasive sales guides

## Features

- Simple chat interface to input product names
- Support for both OpenAI and Ollama LLM providers
- Verbose mode to see detailed agent activities
- Robust error handling with retry mechanisms
- Automatic saving of generated sales guides

## Requirements

- Python 3.8+
- Dependencies listed in `requirements.txt`

## Installation

### Automatic Installation (Recommended)

Choose one of the following setup scripts based on your operating system:

#### For all platforms (using Python):

```bash
python setup.py
```

#### For macOS/Linux (using Bash):

```bash
chmod +x setup.sh  # Make the script executable
./setup.sh
```

#### For Windows (using Batch):

```bash
setup.bat
```

Each script will:
1. Check if Python 3.8+ is installed
2. Create a virtual environment
3. Install all required dependencies
4. Provide instructions for activating the environment and running the project

### Manual Installation

If you prefer to set up manually:

1. Clone this repository or download the files

2. Create and activate a virtual environment:

```bash
# Create a virtual environment
python -m venv venv

# Activate the virtual environment
# On Windows
venv\Scripts\activate
# On macOS/Linux
source venv/bin/activate
```

3. Install the required dependencies:

```bash
pip install -r requirements.txt
```

4. If using OpenAI, make sure you have an API key ready

5. To deactivate the virtual environment when finished:

```bash
# On Windows/macOS/Linux
deactivate
```

## Usage

Run the script with:

```bash
python crew-1.py
```

Follow the prompts to:
1. Enter the product name to research
2. Select the LLM provider (OpenAI or Ollama)
3. Choose whether to enable verbose mode

The system will:
- Search for information about the product
- Scrape relevant websites for detailed data
- Analyze the data for inconsistencies
- Generate a comprehensive sales guide
- Save the guide to a text file

## Error Handling

The system includes robust error handling:
- Retry mechanisms for network errors
- Timeout management for external operations
- Graceful degradation when complete information is unavailable
- Structured error reporting

## Output

The final output is a comprehensive sales guide saved to a text file named after the product (e.g., `iphone_14_sales_guide.txt`).
