#!/usr/bin/env python3
import argparse
import logging
from datetime import datetime
from tradingagents.graph.trading_graph import TradingAgentsGraph
from tradingagents.default_config import DEFAULT_CONFIG
from prompts import STOCK_LIST, system_instruction
from google import genai
import pandas as pd
from io import StringIO
from tradingagents.dataflows.config import get_config, set_config
from tradingagents.dataflows.google import get_fundamentals_google, get_google_news, get_global_news_google
from tradingagents.dataflows.openai import get_global_news_openai
from tradingagents.dataflows.local import get_reddit_global_news
from tradingagents.dataflows.alpha_vantage import get_news as get_alpha_vantage_news
from tradingagents.dataflows.alpha_vantage import get_fundamentals as get_alpha_vantage_fundamentals
from tradingagents.dataflows.alpha_vantage import get_income_statement

import os
import matplotlib.pyplot as plt
import numpy as np

from dotenv import load_dotenv
load_dotenv()

def find_stock_candidates():
    """
    Find stock candidates based on predefined criteria using Google Gemini API.
    
    Returns:
        DataFrame containing stock candidates
    """
    # Use the STOCK_LIST prompt to get stock candidates
    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY or GEMINI_API_KEY environment variable not set")
    config = get_config()
    tools = [
        genai.types.Tool(
            google_search={} # This replaces google_search_retrieval
        )
    ]
    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model="gemini-3-flash-preview",
        config=genai.types.GenerateContentConfig(
            system_instruction=system_instruction,
            tools=tools, # Pass the corrected tool list here
            temperature=0.0
        ),
        contents=STOCK_LIST
    )
    try:
        # Extract text from the response using the correct accessor
        response_text = response.candidates[0].content.parts[0].text
        # Convert the CSV-formatted string to a pandas DataFrame
        df = pd.read_csv(StringIO(response_text))
    except Exception as e:
        raise ValueError(f"Failed to parse stock candidates CSV: {e} {response.text}")
    
    # Remove leading and trailing whitespaces from column names
    df.columns = df.columns.str.strip()
    return df


def plot_stock_analysis(stocks_df):

    # Create visualization: Revenue Growth vs P/E Ratio
    # Remove any rows with missing data
    plot_df = stocks_df.dropna(subset=['P/E Ratio', 'Revenue Growth %'])
    # Create the scatter plot
    fig, ax = plt.subplots(figsize=(14, 10))
    scatter = ax.scatter(plot_df['P/E Ratio'], 
                        plot_df['Revenue Growth %'],
                        s=100, alpha=0.6, c=plot_df['Revenue Growth %'],
                        cmap='RdYlGn', edgecolors='black', linewidth=0.5)
    
    # Add labels for each point
    for idx, row in plot_df.iterrows():
        ax.annotate(row['Ticker'], 
                    (row['P/E Ratio'], row['Revenue Growth %']),
                    xytext=(5, 5), textcoords='offset points',
                    fontsize=8, alpha=0.8)
    
    # Add reference lines for median values
    median_pe = plot_df['P/E Ratio'].median()
    median_growth = plot_df['Revenue Growth %'].median()
    ax.axvline(median_pe, color='blue', linestyle='--', alpha=0.3, label=f'Median P/E: {median_pe:.1f}')
    ax.axhline(median_growth, color='green', linestyle='--', alpha=0.3, label=f'Median Growth: {median_growth:.1f}%')
    
    # Highlight the best value-to-growth quadrant (high growth, low P/E)
    ax.axvspan(0, median_pe, ymin=0.5, ymax=1, alpha=0.05, color='green', label='High Growth, Low P/E')
    
    # Labels and title
    ax.set_xlabel('P/E Ratio', fontsize=12, fontweight='bold')
    ax.set_ylabel('Revenue Growth %', fontsize=12, fontweight='bold')
    ax.set_title('Revenue Growth vs P/E Ratio - Value-to-Growth Analysis', 
                fontsize=14, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right')
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Revenue Growth %', rotation=270, labelpad=20)
    
    # Annotate quadrants
    ax.text(0.02, 0.98, 'Best Value\n(High Growth, Low P/E)', 
            transform=ax.transAxes, fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))
    
    plt.tight_layout()
    
    # Save the plot
    output_filename = 'revenue_growth_vs_pe_analysis.png'
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"\nVisualization saved as: {output_filename}")
    
    # Show the plot
    plt.show()


def find_best_stock_opportunities(stocks_df):
    # Calculate median values
    median_pe = stocks_df['P/E Ratio'].median()
    median_growth = stocks_df['Revenue Growth %'].median()

    # Print outliers analysis
    print("\n=== VALUE-TO-GROWTH OUTLIERS ===")
    print("\nBest Opportunities (High Growth, Low P/E):")
    # Filter for best opportunities: high growth, low P/E
    best_opportunities = stocks_df[
        (stocks_df['Revenue Growth %'] > median_growth) &
        (stocks_df['P/E Ratio'] < median_pe)
    ].sort_values('Revenue Growth %', ascending=False)
    print(best_opportunities[['Ticker', 'Company Name', 'Revenue Growth %', 'P/E Ratio', 'Market Cap']])

    return best_opportunities


def analyze_stock_with_llm(ta, ticker, curr_date):
    
    print(f"Analyzing {ticker} on {curr_date}...")
    
    # forward propagate
    _, decision = ta.propagate(ticker.upper(), curr_date)
    return decision

if __name__ == "__main__":
    # Set up command line argument parser
    parser = argparse.ArgumentParser(description='Run trading analysis on a stock ticker')
    parser.add_argument('--ticker', '-t', type=str, default='',
                        help='Stock ticker symbol to analyze (default: NVDA)')
    parser.add_argument('--date', '-d', type=str, default=datetime.now().strftime('%Y-%m-%d'),
                        help='Analysis date in YYYY-MM-DD format (default: today)')
    parser.add_argument('--find-candidates', '-f', action='store_true',
                        help='Find stock candidates first using LLM before analysis')
    parser.add_argument('--test', action='store_true',
                        help='Run in test mode with mock data')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Enable verbose logging (DEBUG level)')
    parser.add_argument('--list-models', action='store_true',
                        help='List available Gemini models and exit')
    
    args = parser.parse_args()
    
    # Handle list-models flag
    if args.list_models:
        api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        if not api_key:
            print("Error: GOOGLE_API_KEY or GEMINI_API_KEY environment variable not set")
            exit(1)
        client = genai.Client(api_key=api_key)
        print("Available Gemini Models:")
        print("-" * 50)
        for model in client.models.list():
            print(f"Model ID: {model.name}")
            if 'embedContent' in model.supported_actions:
                print(f"Compatible Model: {model.name}")
        exit(0)
    
    # Configure logging level based on verbose flag
    # Only configure tradingagents logger, don't affect other libraries
    tradingagents_logger = logging.getLogger("tradingagents")
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(filename)s:%(lineno)d - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    tradingagents_logger.addHandler(handler)
    tradingagents_logger.setLevel(logging.DEBUG if args.verbose else logging.INFO)
    
    # Create a custom config
    config = DEFAULT_CONFIG.copy()
    # Use Google Gemini models
    config["llm_provider"] = "google"
    config["deep_think_llm"] = "gemini-3-pro-preview"  # Use a different model
    config["quick_think_llm"] = "gemini-2.5-flash"  # Use a different model
    config["max_debate_rounds"] = 1  # Increase debate rounds
    config["backend_url"] = ""

    # Use Aliyun Dashscope LLMs
    #config["llm_provider"] = "QWEN"
    #config["deep_think_llm"] = "qwen-plus"  # Use a different model
    #config["quick_think_llm"] = "qwen-2.5-flash"  # Use a different model
    #config["max_debate_rounds"] = 1  # Increase debate rounds
    #config["backend_url"] = "https://dashscope.aliyuncs.com/compatible-mode/v1"

    # Configure data vendors (default uses yfinance and Alpha Vantage)
    config["data_vendors"] = {
        "core_stock_apis": "alpha_vantage",           # Options: yfinance, alpha_vantage, local
        "technical_indicators": "yfinance",      # Options: yfinance, alpha_vantage, local
        "fundamental_data": "alpha_vantage",     # Options: openai, alpha_vantage, local
        "news_data": "google",            # Options: openai, alpha_vantage, google, local
    }

    # If find_candidates flag is set, generate candidates and exit
    if args.test:
        set_config(config)
        #data = get_fundamentals_google('AAPL', '2026-01-01')
        #data = get_global_news_google('2026-01-31', 7)
        #data = get_google_news('AMZN', '2026-01-31', '2026-02-07')
        #data = get_alpha_vantage_news('AMZN', '2026-01-31', '2026-02-07')
        #data = get_alpha_vantage_fundamentals('AAPL', '2026-01-01')
        data = get_income_statement('AAPL')
        print(data)
        
    elif args.find_candidates:
        stocks_df = find_stock_candidates()
        print("Generated Stock Candidates:")
        print(stocks_df)
        print(f"\nDataFrame shape: {stocks_df.shape}")
        print(f"Columns: {stocks_df.columns.tolist()}")
        find_best_stock_opportunities(stocks_df)
        plot_df = stocks_df.dropna(subset=['Revenue Growth %', 'P/E Ratio'])
        plot_stock_analysis(plot_df)
    else:    

        # Initialize with custom config
        ta = TradingAgentsGraph(debug=args.verbose, config=config)
        #selected_analysts=['social'],

        if args.ticker == "":
            results = {}
            #for ticker, name in list_of_stocks[5:10]:
            for ticker in ['WMT', 'AAPL', 'GOOGL', 'AMZN', 'CSCO']:
                decision = analyze_stock_with_llm(ta, ticker, args.date)
                results[ticker] = decision
            print("Analysis completed for all candidate stocks.")
            print(results)
        else:
            decision = analyze_stock_with_llm(ta, args.ticker, args.date)
            print(f"Final Decision for {args.ticker.upper()} on {args.date}: {decision}")