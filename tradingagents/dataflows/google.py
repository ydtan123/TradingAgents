from typing import Annotated
from datetime import datetime
from dateutil.relativedelta import relativedelta
from .googlenews_utils import getNewsData
import google.generativeai as genai
import os
from .config import get_config


def get_google_news(
    query: Annotated[str, "Query to search with"],
    curr_date: Annotated[str, "Curr date in yyyy-mm-dd format"],
    look_back_days: Annotated[int, "how many days to look back"],
) -> str:
    #import ipdb; ipdb.set_trace()
    # Ensure query is a string and normalize spaces for URL searches
    query = str(query).replace(" ", "+")

    start_date = datetime.strptime(curr_date, "%Y-%m-%d")
    before = start_date - relativedelta(days=7) #look_back_days)
    before = before.strftime("%Y-%m-%d")

    news_results = getNewsData(query, before, curr_date)

    news_str = ""

    for news in news_results:
        print(f"**********{news}")
        news_str += (
            f"### {news['title']} (source: {news['source']}) \n\n{news['snippet']}\n\n"
        )

    if len(news_results) == 0:
        return ""

    return f"## {query} Google News, from {before} to {curr_date}:\n\n{news_str}"


def get_fundamentals_google(ticker, curr_date):
    """
    Get fundamental data for a stock using Google's Gemini API with web search.
    
    Args:
        ticker: Stock ticker symbol
        curr_date: Current date in yyyy-mm-dd format
        
    Returns:
        String containing fundamental analysis data
    """
    config = get_config()
    
    # Configure Google Generative AI
    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY or GEMINI_API_KEY environment variable not set")
    
    genai.configure(api_key=api_key)
    
    # Calculate date range (three months before to current date)
    start_date = datetime.strptime(curr_date, "%Y-%m-%d")
    before = start_date - relativedelta(months=3)
    before_str = before.strftime("%Y-%m-%d")
    
    # Create the model with web search capability
    model = genai.GenerativeModel(
        model_name=config.get("quick_think_llm", "gemini-2.0-flash-exp"))

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

Format the response as markdown with clear sections."""
    
    # Generate content with web search
    response = model.generate_content(prompt)
    
    return response.text

