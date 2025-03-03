#!/bin/bash
# Installation and validation script for NBA Query Tool

echo "=== NBA Query Tool Installation ==="
echo

# Check Python version
echo "Checking Python version..."
python_version=$(python3 --version 2>&1)
if [[ $? -ne 0 ]]; then
  echo "Error: Python 3 not found. Please install Python 3.8 or higher."
  exit 1
fi

echo "Found $python_version"
# Extract version number and check if it's >= 3.8
version=$(echo $python_version | sed 's/Python //g')
major=$(echo $version | cut -d. -f1)
minor=$(echo $version | cut -d. -f2)

if [[ $major -lt 3 || ($major -eq 3 && $minor -lt 8) ]]; then
  echo "Error: Python 3.8 or higher is required."
  exit 1
fi

# Install dependencies
echo
echo "Installing dependencies..."
pip install -r requirements.txt

if [[ $? -ne 0 ]]; then
  echo "Error: Failed to install dependencies."
  exit 1
fi

echo "Dependencies installed successfully!"

# Check for Anthropic API key
echo
echo "Checking for Anthropic API key..."
if [[ -z "$ANTHROPIC_API_KEY" && ! -f .env ]]; then
  echo "Warning: Anthropic API key not found in environment or .env file."
  echo "Would you like to enter your API key now? (y/n)"
  read answer
  if [[ "$answer" == "y" || "$answer" == "Y" ]]; then
    echo "Enter your Anthropic API key:"
    read api_key
    echo "ANTHROPIC_API_KEY=$api_key" > .env
    echo "API key saved to .env file."
  else
    echo "You will need to set the ANTHROPIC_API_KEY environment variable or create a .env file before running the tool."
  fi
else
  if [[ ! -z "$ANTHROPIC_API_KEY" ]]; then
    echo "Found Anthropic API key in environment variables."
  elif [[ -f .env ]]; then
    echo "Found .env file with API key configuration."
  fi
fi

# Check for database file
echo
echo "Checking for NBA database..."
if [[ ! -f "nba.sqlite" ]]; then
  echo "Warning: NBA database (nba.sqlite) not found in current directory."
  echo "You will need to specify the path to the database using the --db option when running the tool."
else
  echo "Found NBA database file."
  
  # Verify it's a valid SQLite database
  if command -v sqlite3 &> /dev/null; then
    sqlite3 nba.sqlite "SELECT name FROM sqlite_master WHERE type='table' LIMIT 1;" &> /dev/null
    if [[ $? -ne 0 ]]; then
      echo "Warning: The file nba.sqlite may not be a valid SQLite database."
    else
      echo "Database validation successful."
    fi
  else
    echo "Note: sqlite3 command not found. Skipping database validation."
  fi
fi

# Make script executable
chmod +x nba_query_tool.py

echo
echo "=== Installation Complete ==="
echo
echo "To run the tool in interactive mode:"
echo "  ./nba_query_tool.py --db path/to/nba.sqlite"
echo
echo "To process a question:"
echo "  ./nba_query_tool.py --db path/to/nba.sqlite --question \"What are the 5 oldest teams in the NBA?\""
echo
echo "To enable automatic query refinement for complex questions:"
echo "  ./nba_query_tool.py --db path/to/nba.sqlite --refine"
echo
echo "For more information, see README.md" 