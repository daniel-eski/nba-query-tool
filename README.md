# NBA Query Tool

A command-line tool that uses Claude 3.7 Sonnet to convert natural language questions about NBA data into SQL queries and execute them against an NBA SQLite database.

## Features

- Convert natural language questions to SQL queries
- Execute queries against an NBA SQLite database
- Display results in a formatted table
- Provide confidence scores to indicate potential inaccuracies
- **Auto-refine low-confidence queries** for improved accuracy
- Support for both interactive and command-line usage
- Colored output for better readability
- Export results to JSON

## Prerequisites

- Python 3.8 or higher
- Anthropic API key
- NBA SQLite database file

## Installation

### Automatic Installation (Recommended)

We provide an installation script that handles dependency installation, API key setup, and basic validation:

```bash
# Clone the repository or download the source code
git clone https://github.com/yourusername/nba-query-tool.git
cd nba-query-tool

# Run the installation script
./install.sh
```

The script will:
- Check your Python version
- Install required dependencies
- Help set up your Anthropic API key
- Check for the NBA database
- Make the tool executable

### Manual Installation

If you prefer to install manually:

1. Clone this repository or download the source code.

2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

3. Set up your Anthropic API key:

   - Option 1: Set the environment variable:
     ```bash
     export ANTHROPIC_API_KEY=your_api_key_here
     ```
   
   - Option 2: Create a `.env` file in the project directory with:
     ```
     ANTHROPIC_API_KEY=your_api_key_here
     ```

4. Make the script executable:
   ```bash
   chmod +x nba_query_tool.py
   ```

## Usage

### Interactive Mode

Run the tool in interactive mode to input questions directly:

```bash
./nba_query_tool.py --db path/to/nba.sqlite
```

With query refinement enabled:

```bash
./nba_query_tool.py --db path/to/nba.sqlite --refine
```

### Command Line Mode

Run the tool with a specific question:

```bash
./nba_query_tool.py --db path/to/nba.sqlite --question "What are the 5 oldest teams in the NBA?"
```

Enable query refinement with custom settings:

```bash
./nba_query_tool.py --db path/to/nba.sqlite --question "Who has scored the most points in NBA history?" --refine --refine-threshold 4 --max-refinements 3
```

Export results to a JSON file:

```bash
./nba_query_tool.py --db path/to/nba.sqlite --question "What are the 5 oldest teams in the NBA?" --output results.json
```

### Command Line Arguments

- `--db`: Path to the NBA SQLite database file (default: "nba.sqlite")
- `--api-key`: Anthropic API key (if not set in environment)
- `--question`, `-q`: Natural language question to process (non-interactive mode)
- `--output`, `-o`: Output file to save results as JSON (optional)
- `--refine`, `-r`: Enable query refinement for low confidence queries
- `--refine-threshold`: Confidence threshold below which to attempt refinement (1-5, default: 3)
- `--max-refinements`: Maximum number of refinement attempts (default: 2)

## Query Refinement

When enabled, the tool can automatically refine queries that have low confidence scores:

1. The tool first generates an initial SQL query based on your question
2. It assesses the confidence in the query's accuracy (1-5 scale)
3. If confidence is below the threshold (default: 3), it attempts to refine the query
4. The refinement process uses the confidence assessment feedback to improve the query
5. This process can repeat up to the maximum refinement attempts (default: 2)

This feature significantly improves results for complex questions that might require multiple iterations to get right.

## Examples

Here are some example questions you can ask:

1. "What are the 5 oldest teams in the NBA?"
2. "Which player scored the most points in a single game?"
3. "What's the average weight of centers?"
4. "How many games went to overtime last season?"
5. "Which team has the highest win percentage at home?"
6. "List all players from Spain"

## Confidence Score

The tool provides a confidence score for each query, ranging from 1-5:

- **1-2**: Low confidence - Results may be incorrect or incomplete
- **3**: Medium confidence - Results may be partially correct
- **4-5**: High confidence - Results are likely correct

## Troubleshooting

- If you encounter a "Database not found" error, make sure the path to the SQLite database is correct.
- If you get authentication errors, check your Anthropic API key.
- For query execution errors, the specific SQLite error will be displayed to help diagnose the issue.
- If you get dependency errors, run `pip install -r requirements.txt` to install all required packages.
- If the tool still doesn't work, try running the installation script: `./install.sh`

## License

This tool is for educational and demonstration purposes only. 
