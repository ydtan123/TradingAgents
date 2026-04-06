from typing import Annotated
from datetime import datetime
from dateutil.relativedelta import relativedelta
from .googlenews_utils import getNewsData
from google import genai
from google.genai import types
import os
from .config import get_config

def search_google_news(prompt: str):
    """
    Search Google News using Gemini API with web search tool.
    
    Args:
        prompt: The search prompt string.
        
    Returns:
        The text response from the Gemini API.
    """
    # Configure Google Generative AI
    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY or GEMINI_API_KEY environment variable not set")
    
    client = genai.Client(api_key=api_key)
    tools = [
        types.Tool(
            google_search={} # This replaces google_search_retrieval
        )
    ]
    system_instruction = """## Persona
You are a Senior Equity Research Analyst with 20 years of experience at a Tier-1 investment bank.
Your goal is to provide objective, data-driven, and high-fidelity financial analysis.
"""
    response = client.models.generate_content(
        model="gemini-3-flash-preview",
        config=types.GenerateContentConfig(
            system_instruction=system_instruction,
            tools=tools
        ),
        contents=prompt
    )
    return response.text

def get_google_news(
    query: Annotated[str, "Query to search with"],
    start_date: Annotated[str, "Start date in yyyy-mm-dd format"],
    end_date: Annotated[str, "End date in yyyy-mm-dd format"]
) -> str:
    start_date_str = datetime.strptime(start_date, "%Y-%m-%d")
    end_date_str = datetime.strptime(end_date, "%Y-%m-%d")
    prompt = f"Search google to collect financial news for {query} from {start_date_str} to {end_date_str}"
    response = search_google_news(prompt)
    return response

def get_global_news_google(
    curr_date: Annotated[str, "Curr date in yyyy-mm-dd format"],
    look_back_days: Annotated[int, "how many days to look back"] = 7,
    limit: Annotated[int, "limit number of articles"] = 5
    ) -> str:
    """
    Get news using Google's Gemini API with web search.
        
    Returns:
        String containing fundamental analysis data
    """
    
    # Configure Google Generative AI
    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY or GEMINI_API_KEY environment variable not set")
    
    # Calculate date range (three months before to current date)
    curr_date_str = datetime.strptime(curr_date, "%Y-%m-%d")
    
    client = genai.Client(api_key=api_key)
    tools = [
        types.Tool(
            google_search={} # This replaces google_search_retrieval
        )
    ]
    system_instruction = """## Persona
You are a Senior Equity Research Analyst with 20 years of experience at a Tier-1 investment bank.
Your goal is to provide objective, data-driven, and high-fidelity financial analysis.
"""
    
    prompt =  f"""Can you search global or macroeconomics news from {look_back_days} days 
    before {curr_date_str} to {curr_date_str} that would be informative for trading purposes? 
    Make sure you only get the data posted during that period. Limit the results to {limit} 
    articles."""

    response = client.models.generate_content(
        model="gemini-3-flash-preview",
        config=types.GenerateContentConfig(
            system_instruction=system_instruction,
            tools=tools, # Pass the corrected tool list here
            temperature=0.0
        ),
        contents=prompt
    )
    
    return response.text


def get_fundamentals_google(ticker, curr_date):
    """
    Get fundamental data for a stock using Google's Gemini API with web search.
    
    Args:
        ticker: Stock ticker symbol
        curr_date: Current date in yyyy-mm-dd format
        
    Returns:
        String containing fundamental analysis data
    """
    
    # Configure Google Generative AI
    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY or GEMINI_API_KEY environment variable not set")
    
    # Calculate date range (three months before to current date)
    start_date = datetime.strptime(curr_date, "%Y-%m-%d")
    before = start_date - relativedelta(months=6)
    before_str = before.strftime("%Y-%m-%d")
    
    # Create the prompt
    prompt = f"""Can you search for fundamental data and discussions on {ticker} from {before_str} to {curr_date}. 
Make sure you only get the data posted during that period. 
Please provide the information in a table format, including key metrics such as:
- P/E Ratio (Price-to-Earnings)
- P/S Ratio (Price-to-Sales)
- P/B Ratio (Price-to-Book)
- Cash Flow metrics
- Revenue
- Earnings
- Debt-to-Equity
- Market Cap
- Any other relevant fundamental metrics

Format the response as markdown with clear sections.
Search the web to get the most recent and accurate data.
"""
    
    # Generate content with web search
    config = get_config()
    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model=config.get("quick_think_llm", "gemini-2.5-flash"),
        contents=prompt)
    
    return response.text

