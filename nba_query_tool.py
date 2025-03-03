#!/usr/bin/env python3
# nba_query_tool.py - A command-line tool to convert natural language questions about NBA to SQL queries
# and execute them against the nba.sqlite database.

import os
import time
import random
import sqlite3
import re
import json
import sys
import argparse
import textwrap
from typing import List, Tuple, Dict, Any, Optional
import logging

# Check for required dependencies
def check_dependencies():
    missing = []
    
    try:
        import anthropic
    except ImportError:
        missing.append("anthropic")
    
    try:
        import tabulate
    except ImportError:
        missing.append("tabulate")
    
    try:
        import colorama
    except ImportError:
        missing.append("colorama")
    
    if missing:
        print("Error: Missing required dependencies:")
        print("\n".join(f"  - {pkg}" for pkg in missing))
        print("\nPlease install the required dependencies:")
        print("pip install -r requirements.txt")
        sys.exit(1)

# Run dependency check
check_dependencies()

# Now import dependencies
from anthropic import Anthropic
from tabulate import tabulate
from colorama import init, Fore, Style

# Try to load .env file if dotenv is available
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv is optional

# Initialize colorama for cross-platform colored terminal output
init()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("nba_query_tool")

class AnthropicRateLimiter:
    """
    Helper class to handle Anthropic API rate limits with exponential backoff and retries
    """
    def __init__(self, max_retries=5, initial_backoff=2, backoff_factor=2, jitter=0.1):
        self.max_retries = max_retries
        self.initial_backoff = initial_backoff
        self.backoff_factor = backoff_factor
        self.jitter = jitter
        self.last_request_time = 0
        self.min_request_interval = 0.5  # Minimum time between requests in seconds

    def call_with_retry(self, api_func, *args, **kwargs):
        """
        Call an API function with retry logic for rate limits
        """
        # Ensure we're not making too many requests too quickly
        self._throttle()
        
        retry_count = 0
        while retry_count <= self.max_retries:
            try:
                # Make the API call
                response = api_func(*args, **kwargs)
                self.last_request_time = time.time()
                return response
            except Exception as e:
                error_message = str(e).lower()
                
                # Check if it's a rate limit error
                if "rate limit" in error_message or "too many requests" in error_message:
                    retry_count += 1
                    if retry_count > self.max_retries:
                        logger.error(f"Maximum retries ({self.max_retries}) exceeded. Giving up.")
                        raise
                    
                    # Calculate backoff time with exponential backoff and jitter
                    backoff_time = self.initial_backoff * (self.backoff_factor ** (retry_count - 1))
                    jitter_amount = backoff_time * self.jitter * random.uniform(-1, 1)
                    sleep_time = backoff_time + jitter_amount
                    
                    logger.warning(f"Rate limit exceeded. Retrying in {sleep_time:.2f} seconds (retry {retry_count}/{self.max_retries})")
                    time.sleep(sleep_time)
                else:
                    # Not a rate limit error, re-raise it
                    logger.error(f"API error: {error_message}")
                    raise
    
    def _throttle(self):
        """Ensure we don't exceed the rate limit by adding delay if needed"""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.min_request_interval:
            sleep_time = self.min_request_interval - elapsed
            time.sleep(sleep_time)

class NBAQueryTool:
    """
    Tool to convert natural language questions about NBA to SQL queries
    and execute them against the NBA database.
    """
    def __init__(self, db_path: str, api_key: Optional[str] = None):
        self.db_path = db_path
        
        # Validate database file
        self._validate_database()
        
        # Get API key from environment if not provided
        if not api_key:
            api_key = os.environ.get("ANTHROPIC_API_KEY")
            if not api_key:
                raise ValueError("Anthropic API key must be provided or set as ANTHROPIC_API_KEY environment variable")
        
        # Initialize Anthropic client
        self.client = Anthropic(api_key=api_key)
        self.rate_limiter = AnthropicRateLimiter()
        
        # Load database schema
        self.db_schema = self._get_database_schema()
        
        # Setup prompt
        self.system_prompt = self._setup_prompt()
    
    def _validate_database(self):
        """Validate that the database file exists and is a valid SQLite database"""
        if not os.path.exists(self.db_path):
            raise FileNotFoundError(f"Database file not found: {self.db_path}")
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Check if this is a valid SQLite database by running a simple query
            cursor.execute("SELECT sqlite_version();")
            conn.close()
        except sqlite3.Error as e:
            raise ValueError(f"Invalid SQLite database: {e}")
    
    def _get_database_schema(self) -> str:
        """
        Extract database schema from the SQLite database.
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get list of all tables
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = cursor.fetchall()
            
            schema = []
            for table in tables:
                table_name = table[0]
                cursor.execute(f"PRAGMA table_info({table_name});")
                columns = cursor.fetchall()
                
                column_list = [f"{col[1]} ({col[2]})" for col in columns]
                schema.append(f"Table: {table_name}")
                schema.append("Columns: " + ", ".join(column_list))
                schema.append("")  # Empty line for readability
            
            conn.close()
            return "\n".join(schema)
        except Exception as e:
            logger.error(f"Error getting database schema: {str(e)}")
            return "Error retrieving schema"
    
    def _setup_prompt(self) -> str:
        """
        Setup the prompt for SQL generation
        """
        system_prompt = """You are an expert system designed to translate natural language questions about NBA statistics into precise SQL queries. Your task is to analyze each question carefully and generate a valid SQL query that will extract the requested information from the NBA statistics database.

The following schema represents the tables and columns available in the NBA database. Only use tables and column names that appear in this schema:
{schema}

To convert a natural language question to SQL, follow these steps (1-5):
1. Identify the Question Type:
Counting: Queries that ask "how many" or require counting entities
Filtering: Queries that ask for lists of entities matching specific criteria
Ranking: Queries that ask for top/bottom N entities
Aggregation: Queries that calculate averages, sums, or percentages
Comparison: Queries that compare different groups or conditions
Detail: Queries that ask for specific information about an entity
History: Queries that involve historical data or changes over time


2. Select Tables and Aliases:
Identify the primary tables needed and assign standard aliases:
team → t
game → g
player → p
common_player_info → cpi
draft_history → dh
game_info → gi
line_score → ls
other_stats → os
officials → o
team_details → td
team_info_common → tic


3. Apply SQL Patterns Based on Question Type:
For counting queries:
Use COUNT(*) as [entity]_count
Always add LIMIT 1 at the end
Example: SELECT COUNT(*) as team_count FROM team LIMIT 1
 For filtering queries:
Select specific columns (not *)
Use clear WHERE conditions
No LIMIT unless specifically requested
Example: SELECT first_name, last_name FROM common_player_info WHERE country = 'Spain'
 For ranking queries:
Include ORDER BY with appropriate direction (DESC/ASC)
Add LIMIT with the requested number
Example: SELECT full_name FROM team ORDER BY year_founded ASC LIMIT 5
 For aggregation queries:
Use ROUND() for decimal values, typically to 2 places
Always add LIMIT 1 when returning a single aggregated value
Example: SELECT ROUND(AVG(pts_home + pts_away), 2) as avg_points FROM game LIMIT 1
 For percentage calculations:
Use consistent format: CAST(numerator AS FLOAT) / denominator * 100
Combine with ROUND() for clean display
Example: ROUND(CAST(SUM(CASE WHEN condition THEN 1 ELSE 0 END) AS FLOAT) / COUNT(*) * 100, 2)
 For queries involving absence/exclusion (never, not, etc.):
Use NOT EXISTS with a correlated subquery pattern
Example: SELECT t.full_name FROM team t WHERE NOT EXISTS (SELECT 1 FROM game g WHERE condition AND (g.team_id_home = t.id OR g.team_id_away = t.id))
 For date handling:
Always first seek to reference game_info table instead of game table where possible when handling analysis involving dates
Use strftime('%Y', date_column) for year extraction
Use strftime('%m-%d', date_column) for month-day extraction
Example: SELECT COUNT(DISTINCT game_id) as games_played FROM game_info WHERE strftime('%Y', game_date) = '2010' for games in the year 2010

For analyzing attendance:
Unlike data handling, always first seek to reference game table instead of game_info table where possible analyzing questions involving game or arena attendance

For string manipulation:
Use SUBSTR(column, start, length) for substring extraction
Use INSTR(column, substring) to find position of substring
Use column1 || ' ' || column2 for concatenation
Example: first_name || ' ' || last_name as full_name

4. Handle Common Patterns:
Team identification - teams can be home or away:
Use t.id IN (g.team_id_home, g.team_id_away) to match either case
Use CASE expressions to handle different columns based on home/away
Example: CASE WHEN g.team_id_home = t.id THEN g.pts_home ELSE g.pts_away END
 Multiple conditions:
Use IN operator instead of multiple OR conditions
Example: WHERE team_name IN ('Lakers', 'Celtics') instead of team_name = 'Lakers' OR team_name = 'Celtics'
 Conditional counting:
Use SUM(CASE WHEN condition THEN 1 ELSE 0 END) pattern
 Filtering with thresholds:
Use HAVING COUNT(*) >= N for reliable statistics
Example: GROUP BY t.id, t.full_name HAVING COUNT(*) >= 100
 Nested logic:
Use subqueries with clear aliases or CTEs with WITH clause
For complex calculations, use CAST to ensure floating-point division


5. Final Formatting:
Use appropriate column aliases with AS for readability
For GROUP BY, include both ID and display columns
Example: GROUP BY t.id, t.full_name
For aggregation results, always use LIMIT 1 if returning just one row
Ensure proper use of quotes: single quotes for strings, no quotes for numbers

Output Format:
Only provide output in a single line of SQL

Reminders:
- Always be extra careful when handling columns where there might be data entry errors (e.g., use clauses to check if a column value is not null and also if column value is > 0)
- Make sure to use proper SQL syntax for SQLite, which may differ slightly from other SQL dialects.
- Include explicit LIMIT clauses when appropriate.
- Use NOT EXISTS for exclusion queries rather than LEFT JOIN with NULL checks.
- When handling percentages, always use CAST to ensure floating-point division.
- For string operations, use proper SQLite functions (SUBSTR, INSTR, ||).

Here are some examples of translating questions to SQL:

Example 1:
Question: "What's the average weight of power forwards?"
SQL: SELECT ROUND(AVG(CAST(weight AS FLOAT)), 2) as avg_weight FROM common_player_info WHERE position LIKE '%F%' AND weight != '' AND weight != 'None' LIMIT 1

Example 2:
Question: "What are the 5 oldest teams in the NBA?"
SQL: SELECT full_name FROM team ORDER BY year_founded ASC LIMIT 5

Example 3:
Question: "List all teams that have never made the playoffs"
SQL: SELECT t.full_name FROM team t WHERE NOT EXISTS (SELECT 1 FROM game g WHERE g.season_type = 'Playoffs' AND (g.team_id_home = t.id OR g.team_id_away = t.id))

Example 4:
Question: "What's the average game time in hours?"
SQL: SELECT ROUND(AVG(CAST(SUBSTR(game_time, 1, INSTR(game_time, ':') - 1) AS INTEGER)), 2) as avg_hours FROM game_info WHERE game_time != '' LIMIT 1

Example 5:
Question: "Which team has the biggest home court advantage in terms of win percentage?"
SQL: SELECT t.full_name FROM game g JOIN team t ON t.id IN (g.team_id_home, g.team_id_away) GROUP BY t.id, t.full_name HAVING COUNT(*) >= 100 ORDER BY (CAST(SUM(CASE WHEN g.team_id_home = t.id AND g.pts_home > g.pts_away THEN 1 ELSE 0 END) AS FLOAT) / COUNT(CASE WHEN g.team_id_home = t.id THEN 1 END) * 100 - CAST(SUM(CASE WHEN g.team_id_away = t.id AND g.pts_away > g.pts_home THEN 1 ELSE 0 END) AS FLOAT) / COUNT(CASE WHEN g.team_id_away = t.id THEN 1 END) * 100) DESC LIMIT 1

Example 6:
Question: "What's the win percentage difference between teams with 3+ rest days vs fewer rest days?"
SQL: WITH rest_days AS (SELECT g.game_id, t.id, t.full_name, JULIANDAY(g.game_date) - JULIANDAY(LAG(g.game_date) OVER (PARTITION BY t.id ORDER BY g.game_date)) as days_rest, CASE WHEN (g.team_id_home = t.id AND g.pts_home > g.pts_away) OR (g.team_id_away = t.id AND g.pts_away > g.pts_home) THEN 1 ELSE 0 END as won FROM game g JOIN team t ON t.id IN (g.team_id_home, g.team_id_away)) SELECT ROUND(AVG(CASE WHEN days_rest >= 3 THEN won END) * 100, 2) - ROUND(AVG(CASE WHEN days_rest < 3 THEN won END) * 100, 2) as win_pct_diff FROM rest_days WHERE days_rest IS NOT NULL

Example 7:
Question: "List all games that went to overtime"
SQL: SELECT COUNT(*) as ot_games FROM line_score WHERE pts_ot1_home IS NOT NULL AND pts_ot1_home >0 LIMIT 1

Example 8:
Question: "Which team has the highest percentage of international players?"
SQL: SELECT t.full_name FROM common_player_info cpi JOIN team t ON cpi.team_id = t.id GROUP BY cpi.team_id, t.full_name HAVING COUNT(*) >= 10 ORDER BY CAST(SUM(CASE WHEN cpi.country != 'USA' THEN 1 ELSE 0 END) AS FLOAT) / COUNT(*) * 100 DESC LIMIT 1

Example 9:
Question: "What's the most common final score in overtime games?"
SQL: SELECT g.pts_home || '-' || g.pts_away as score FROM game g JOIN line_score l ON g.game_id = l.game_id WHERE l.pts_ot1_home IS NOT NULL GROUP BY g.pts_home, g.pts_away ORDER BY COUNT(*) DESC LIMIT 1

Example 10:
Question: "How many games were played on New Year's Day?"
SQL: SELECT COUNT(*) FROM game_info WHERE strftime('%m-%d', game_date) = '01-01'"""

        return system_prompt.replace("{schema}", self.db_schema)
    
    def generate_sql(self, question: str) -> Dict[str, Any]:
        """
        Generate SQL from natural language question
        
        Args:
            question: Natural language question
            
        Returns:
            Dictionary with generated SQL and confidence/uncertainty level
        """
        # Prepare user message
        user_content = f"Now generate SQL code to answer this question: \"{question}\"\nRespond with only the SQL query and nothing else."
        
        messages = [{"role": "user", "content": user_content}]
        
        # Call the API with the correct parameters
        try:
            # Create parameters dictionary
            api_params = {
                "model": "claude-3-7-sonnet-20250219",
                "max_tokens": 1000,
                "temperature": 0,
                "system": self.system_prompt,
                "messages": messages
            }
            
            # Make the API call using rate limiter
            def api_call():
                return self.client.messages.create(**api_params)
            
            print(f"{Fore.YELLOW}Generating SQL query...{Style.RESET_ALL}")
            response = self.rate_limiter.call_with_retry(api_call)
            
            # Get the raw text from the response
            raw_text = response.content[0].text.strip()
            
            # Extract SQL query from the response
            sql = self._extract_sql_from_text(raw_text)
            sql = sql.strip()
            
            # Get uncertainty assessment by using the model's certainty assessment
            api_params_certainty = {
                "model": "claude-3-7-sonnet-20250219",
                "max_tokens": 500,
                "temperature": 0,
                "system": f"""You are an expert SQL evaluator that assesses the confidence in a generated SQL query's ability to correctly answer a natural language question. 
                
Assess on a scale of 1-5 where 1 means very low confidence (likely incorrect) and 5 means very high confidence (almost certainly correct).

IMPORTANT: The database has the following schema. Queries should ONLY use tables and columns that appear in this schema:
{self.db_schema}

Provide a single number followed by a brief explanation. If the query uses tables or columns that don't exist in the schema, the confidence should be 1.""",
                "messages": [
                    {"role": "user", "content": f"Question: {question}\nGenerated SQL: {sql}\n\nRate your confidence that this SQL query will correctly answer the question (1-5):"}
                ]
            }
            
            def certainty_api_call():
                return self.client.messages.create(**api_params_certainty)
            
            print(f"{Fore.YELLOW}Evaluating query confidence...{Style.RESET_ALL}")
            certainty_response = self.rate_limiter.call_with_retry(certainty_api_call)
            certainty_text = certainty_response.content[0].text.strip()
            
            # Extract confidence score (1-5)
            confidence_match = re.search(r'^([1-5])', certainty_text)
            confidence_score = int(confidence_match.group(1)) if confidence_match else 3
            
            # Get explanation without the score
            confidence_explanation = re.sub(r'^[1-5][\s\-:\.]*', '', certainty_text).strip()
            
            return {
                "sql": sql,
                "confidence_score": confidence_score,
                "confidence_explanation": confidence_explanation
            }
            
        except Exception as e:
            logger.error(f"Error generating SQL: {str(e)}")
            return {
                "sql": f"ERROR: {str(e)}",
                "confidence_score": 0,
                "confidence_explanation": "Failed to generate SQL due to an error."
            }
    
    def _extract_sql_from_text(self, text: str) -> str:
        """
        Extract SQL query from text, handling various formats
        
        Args:
            text: Text containing SQL query
            
        Returns:
            Extracted SQL query
        """
        # Try to extract SQL from markdown code blocks first
        sql_code_block_patterns = [
            r"```sql\s*(.*?)\s*```",  # SQL code block with sql tag
            r"```\s*(SELECT.*?)\s*```",  # Code block starting with SELECT
            r"```\s*(WITH.*?)\s*```",  # Code block starting with WITH
        ]
        
        for pattern in sql_code_block_patterns:
            matches = re.findall(pattern, text, re.DOTALL)
            if matches:
                return matches[0].strip()
        
        # If no code blocks, try to find SQL statements directly
        sql_patterns = [
            r"(SELECT\s+.*?)(;|\Z)",  # SELECT statement
            r"(WITH\s+.*?)(;|\Z)"    # WITH statement
        ]
        
        for pattern in sql_patterns:
            matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)
            if matches:
                # Return the first group of the first match
                return matches[0][0].strip()
        
        # Return the raw text if no SQL was found
        return text
    
    def execute_sql(self, sql: str) -> Dict[str, Any]:
        """
        Execute SQL query against the database
        
        Args:
            sql: SQL query to execute
            
        Returns:
            Dictionary with query results, column names, and execution status
        """
        # Check for empty or error SQL
        if not sql or sql.startswith("ERROR"):
            return {
                "success": False,
                "error": sql if sql else "Empty SQL query",
                "results": [],
                "column_names": []
            }
            
        try:
            # Add a safety limit if not already present to prevent runaway queries
            if "limit" not in sql.lower():
                # Only add limit for SELECT queries
                if sql.strip().lower().startswith("select"):
                    sql = f"{sql} LIMIT 1000"
            
            conn = sqlite3.connect(self.db_path)
            
            # Enable column names in results
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            # Set a timeout to prevent long-running queries
            conn.execute("PRAGMA timeout = 5000")  # 5 second timeout
            
            cursor.execute(sql)
            results = cursor.fetchall()
            
            # Convert row objects to dictionaries
            column_names = [description[0] for description in cursor.description]
            result_dicts = [dict(row) for row in results]
            
            conn.close()
            
            return {
                "success": True,
                "results": result_dicts,
                "column_names": column_names,
                "row_count": len(result_dicts)
            }
        except sqlite3.OperationalError as e:
            error_msg = str(e)
            logger.error(f"SQLite operational error executing SQL: {error_msg}")
            return {
                "success": False,
                "error": f"SQLite operational error: {error_msg}",
                "results": [],
                "column_names": []
            }
        except sqlite3.Error as e:
            error_msg = str(e)
            logger.error(f"SQLite error executing SQL: {error_msg}")
            return {
                "success": False,
                "error": f"SQLite error: {error_msg}",
                "results": [],
                "column_names": []
            }
        except Exception as e:
            logger.error(f"Error executing SQL: {str(e)}")
            return {
                "success": False,
                "error": f"Error: {str(e)}",
                "results": [],
                "column_names": []
            }
    
    def _refine_sql_query(self, question: str, original_sql: str, confidence_score: int, confidence_explanation: str) -> Dict[str, Any]:
        """
        Refine an SQL query based on confidence assessment feedback
        
        Args:
            question: Original natural language question
            original_sql: Original SQL query
            confidence_score: Confidence score (1-5)
            confidence_explanation: Explanation of issues with the query
            
        Returns:
            Dictionary with refined SQL and confidence assessment
        """
        # Prepare system message for query refinement
        system_content = f"""You are an expert SQL engineer that improves SQL queries based on identified issues. 
You will be given a natural language question, an original SQL query, and feedback about 
issues with that query. Your task is to fix all the issues and provide a corrected SQL query 
that will properly answer the original question.

IMPORTANT: The database has the following schema. You MUST ONLY use tables and columns that appear in this schema:
{self.db_schema}

Only output the improved SQL query without explanation."""
        
        # Prepare user message
        user_content = f"""Original question: {question}
Original SQL query: {original_sql}
Confidence score: {confidence_score}/5
Issues identified: {confidence_explanation}

Please provide an improved SQL query that addresses these issues and correctly answers the original question.
IMPORTANT: Only use tables and columns that exist in the database schema.
"""
        
        messages = [{"role": "user", "content": user_content}]
        
        # Call the API with the correct parameters
        try:
            # Create parameters dictionary
            api_params = {
                "model": "claude-3-7-sonnet-20250219",
                "max_tokens": 1000,
                "temperature": 0,
                "system": system_content,
                "messages": messages
            }
            
            # Make the API call using rate limiter
            def api_call():
                return self.client.messages.create(**api_params)
            
            print(f"{Fore.YELLOW}Refining SQL query...{Style.RESET_ALL}")
            response = self.rate_limiter.call_with_retry(api_call)
            
            # Get the raw text from the response
            raw_text = response.content[0].text.strip()
            
            # Extract SQL query from the response
            refined_sql = self._extract_sql_from_text(raw_text)
            refined_sql = refined_sql.strip()
            
            # Get confidence assessment for the refined query
            api_params_certainty = {
                "model": "claude-3-7-sonnet-20250219",
                "max_tokens": 500,
                "temperature": 0,
                "system": f"""You are an expert SQL evaluator that assesses the confidence in a generated SQL query's ability to correctly answer a natural language question. 
                
Assess on a scale of 1-5 where 1 means very low confidence (likely incorrect) and 5 means very high confidence (almost certainly correct).

IMPORTANT: The database has the following schema. Queries should ONLY use tables and columns that appear in this schema:
{self.db_schema}

Provide a single number followed by a brief explanation. If the query uses tables or columns that don't exist in the schema, the confidence should be 1.""",
                "messages": [
                    {"role": "user", "content": f"Question: {question}\nGenerated SQL: {refined_sql}\n\nRate your confidence that this SQL query will correctly answer the question (1-5):"}
                ]
            }
            
            def certainty_api_call():
                return self.client.messages.create(**api_params_certainty)
            
            print(f"{Fore.YELLOW}Evaluating refined query confidence...{Style.RESET_ALL}")
            certainty_response = self.rate_limiter.call_with_retry(certainty_api_call)
            certainty_text = certainty_response.content[0].text.strip()
            
            # Extract confidence score (1-5)
            confidence_match = re.search(r'^([1-5])', certainty_text)
            confidence_score = int(confidence_match.group(1)) if confidence_match else 3
            
            # Get explanation without the score
            confidence_explanation = re.sub(r'^[1-5][\s\-:\.]*', '', certainty_text).strip()
            
            return {
                "sql": refined_sql,
                "confidence_score": confidence_score,
                "confidence_explanation": confidence_explanation
            }
            
        except Exception as e:
            logger.error(f"Error refining SQL: {str(e)}")
            # If refinement fails, return the original query
            return {
                "sql": original_sql,
                "confidence_score": 0,
                "confidence_explanation": f"Failed to refine SQL due to an error: {str(e)}"
            }

    def process_question(self, question: str, refine_threshold: int = 3, max_refinement_attempts: int = 2) -> Dict[str, Any]:
        """
        Process a natural language question about NBA data
        
        Args:
            question: Natural language question
            refine_threshold: Confidence threshold below which to attempt refinement (1-5)
            max_refinement_attempts: Maximum number of refinement attempts
            
        Returns:
            Dictionary with results, including the SQL query and execution results
        """
        # Generate SQL query
        sql_generation = self.generate_sql(question)
        sql_query = sql_generation.get("sql", "")
        confidence_score = sql_generation.get("confidence_score", 0)
        confidence_explanation = sql_generation.get("confidence_explanation", "")
        
        refinement_history = []
        
        # Store original query in refinement history
        refinement_history.append({
            "sql": sql_query,
            "confidence_score": confidence_score,
            "confidence_explanation": confidence_explanation
        })
        
        # Refine the query if confidence is below threshold
        refinement_attempts = 0
        while confidence_score < refine_threshold and refinement_attempts < max_refinement_attempts:
            print(f"{Fore.YELLOW}Low confidence score ({confidence_score}/5). Attempting query refinement...{Style.RESET_ALL}")
            
            # Attempt to refine the query
            refined = self._refine_sql_query(question, sql_query, confidence_score, confidence_explanation)
            
            # Store refinement in history
            refinement_history.append(refined)
            
            # Update current query and confidence
            sql_query = refined.get("sql", sql_query)
            confidence_score = refined.get("confidence_score", confidence_score)
            confidence_explanation = refined.get("confidence_explanation", confidence_explanation)
            
            refinement_attempts += 1
            
            # If we've reached max attempts or confidence is now above threshold, break
            if confidence_score >= refine_threshold:
                print(f"{Fore.GREEN}Query successfully refined! New confidence: {confidence_score}/5{Style.RESET_ALL}")
                break
                
            if refinement_attempts >= max_refinement_attempts:
                print(f"{Fore.RED}Reached maximum refinement attempts. Using best query found.{Style.RESET_ALL}")
        
        # Execute the final SQL query
        execution_result = self.execute_sql(sql_query)
        
        # Combine results
        return {
            "question": question,
            "sql": sql_query,
            "execution_success": execution_result.get("success", False),
            "execution_error": execution_result.get("error", None),
            "results": execution_result.get("results", []),
            "column_names": execution_result.get("column_names", []),
            "row_count": execution_result.get("row_count", 0),
            "confidence_score": confidence_score,
            "confidence_explanation": confidence_explanation,
            "refinement_history": refinement_history,
            "refinement_attempts": refinement_attempts
        }

def display_results(result_data: Dict[str, Any]):
    """Format and display the results of a query"""
    print(f"\n{Fore.CYAN}== Query Results =={Style.RESET_ALL}")
    print(f"{Fore.GREEN}Question:{Style.RESET_ALL} {result_data['question']}")
    
    # Check if there was query refinement
    refinement_history = result_data.get('refinement_history', [])
    refinement_attempts = result_data.get('refinement_attempts', 0)
    
    if refinement_attempts > 0:
        print(f"\n{Fore.MAGENTA}Query Refinement History:{Style.RESET_ALL}")
        
        for i, refinement in enumerate(refinement_history):
            iteration_label = "Original Query" if i == 0 else f"Refinement {i}"
            confidence_score = refinement.get('confidence_score', 0)
            confidence_color = Fore.RED if confidence_score <= 2 else (Fore.YELLOW if confidence_score == 3 else Fore.GREEN)
            
            print(f"\n{Fore.MAGENTA}{iteration_label} (Confidence: {confidence_color}{confidence_score}/5{Style.RESET_ALL}{Fore.MAGENTA}):{Style.RESET_ALL}")
            print(f"{Fore.WHITE}{refinement.get('sql', 'N/A')}{Style.RESET_ALL}")
            
            # Show issues for non-final iterations (skip showing issues for the final result since we'll show it below)
            if i < len(refinement_history) - 1:
                print(f"{Fore.MAGENTA}Issues:{Style.RESET_ALL} {confidence_color}{refinement.get('confidence_explanation', 'None')}{Style.RESET_ALL}")
    
    # Display final SQL query with syntax highlighting or formatting
    print(f"\n{Fore.YELLOW}Final SQL Query:{Style.RESET_ALL}")
    print(f"{Fore.WHITE}{result_data['sql']}{Style.RESET_ALL}")
    
    # Display confidence assessment
    confidence_score = result_data.get('confidence_score', 0)
    confidence_color = Fore.RED if confidence_score <= 2 else (Fore.YELLOW if confidence_score == 3 else Fore.GREEN)
    
    print(f"\n{Fore.BLUE}Confidence Score:{Style.RESET_ALL} {confidence_color}{confidence_score}/5{Style.RESET_ALL}")
    print(f"{Fore.BLUE}Confidence Assessment:{Style.RESET_ALL} {confidence_color}{result_data.get('confidence_explanation', '')}{Style.RESET_ALL}")
    
    # Display execution status
    if result_data['execution_success']:
        print(f"\n{Fore.GREEN}Execution Status:{Style.RESET_ALL} Success")
        print(f"{Fore.GREEN}Row Count:{Style.RESET_ALL} {result_data['row_count']}")
        
        # Display results in a table format if there are any
        if result_data['results']:
            print(f"\n{Fore.GREEN}Results:{Style.RESET_ALL}")
            # Use tabulate for nice table formatting
            table_data = []
            for row in result_data['results']:
                table_data.append([row[col] for col in result_data['column_names']])
            
            print(tabulate(table_data, headers=result_data['column_names'], tablefmt="pretty"))
        else:
            print(f"\n{Fore.YELLOW}No results returned{Style.RESET_ALL}")
    else:
        print(f"\n{Fore.RED}Execution Status:{Style.RESET_ALL} Failed")
        print(f"{Fore.RED}Error:{Style.RESET_ALL} {result_data['execution_error']}")

def interactive_mode(query_tool: NBAQueryTool, refine_threshold: int, max_refinements: int):
    """Run the tool in interactive mode, prompting for questions"""
    print(f"\n{Fore.CYAN}===== NBA Database Query Tool ====={Style.RESET_ALL}")
    print(f"{Fore.CYAN}Enter your questions about NBA data in natural language{Style.RESET_ALL}")
    print(f"{Fore.CYAN}Type 'exit', 'quit', or 'q' to quit{Style.RESET_ALL}\n")
    
    while True:
        try:
            question = input(f"{Fore.GREEN}Question:{Style.RESET_ALL} ")
            question = question.strip()
            
            if question.lower() in ('exit', 'quit', 'q'):
                print(f"{Fore.YELLOW}Goodbye!{Style.RESET_ALL}")
                break
                
            if not question:
                continue
                
            # Process the question
            result = query_tool.process_question(question, refine_threshold=refine_threshold, max_refinement_attempts=max_refinements)
            
            # Display the results
            display_results(result)
            
            print("\n" + "-" * 80 + "\n")
            
        except KeyboardInterrupt:
            print(f"\n{Fore.YELLOW}Interrupted by user. Goodbye!{Style.RESET_ALL}")
            break
        except Exception as e:
            print(f"\n{Fore.RED}Error: {str(e)}{Style.RESET_ALL}")

def main():
    """Main function to run the NBA query tool"""
    parser = argparse.ArgumentParser(description="NBA Database Query Tool")
    parser.add_argument("--db", type=str, default="nba.sqlite", help="Path to the NBA SQLite database")
    parser.add_argument("--api-key", type=str, help="Anthropic API key (if not set in environment)")
    parser.add_argument("--question", "-q", type=str, help="Natural language question to process (non-interactive mode)")
    parser.add_argument("--output", "-o", type=str, help="Output file to save results as JSON (optional)")
    parser.add_argument("--refine", "-r", action="store_true", help="Enable query refinement for low confidence queries")
    parser.add_argument("--refine-threshold", type=int, choices=range(1, 6), default=3, 
                        help="Confidence threshold below which to attempt refinement (1-5, default: 3)")
    parser.add_argument("--max-refinements", type=int, default=2, 
                        help="Maximum number of refinement attempts (default: 2)")
    
    args = parser.parse_args()
    
    try:
        # Create query tool
        query_tool = NBAQueryTool(db_path=args.db, api_key=args.api_key)
        
        # Process arguments for refinement
        refine_threshold = args.refine_threshold if args.refine else 0  # If refinement disabled, set threshold to 0
        max_refinements = args.max_refinements
        
        if args.question:
            # Non-interactive mode
            result = query_tool.process_question(args.question, refine_threshold=refine_threshold, max_refinement_attempts=max_refinements)
            
            # Display results
            display_results(result)
            
            # Save results to output file if specified
            if args.output:
                with open(args.output, 'w') as f:
                    json.dump(result, f, indent=2)
                print(f"\n{Fore.GREEN}Results saved to {args.output}{Style.RESET_ALL}")
        else:
            # Interactive mode
            interactive_mode(query_tool, refine_threshold=refine_threshold, max_refinements=max_refinements)
            
    except FileNotFoundError:
        print(f"{Fore.RED}Error: Database file not found: {args.db}{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}Please make sure the NBA SQLite database file exists at the specified path.{Style.RESET_ALL}")
        sys.exit(1)
    except ValueError as e:
        print(f"{Fore.RED}Error: {str(e)}{Style.RESET_ALL}")
        sys.exit(1)
    except Exception as e:
        print(f"{Fore.RED}Unexpected error: {str(e)}{Style.RESET_ALL}")
        sys.exit(1)

if __name__ == "__main__":
    main() 