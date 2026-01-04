#!/usr/bin/env python3
import argparse
from datetime import datetime
from tradingagents.graph.trading_graph import TradingAgentsGraph
from tradingagents.default_config import DEFAULT_CONFIG
import google.generativeai as genai
import pandas as pd
from io import StringIO
from tradingagents.dataflows.config import get_config
import os
import matplotlib.pyplot as plt
import numpy as np

from dotenv import load_dotenv
load_dotenv()

def find_stock_candidates():
    config = get_config()
    
    # Configure Google Generative AI
    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY or GEMINI_API_KEY environment variable not set")
    

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-3-pro-preview')
    full_prompt = """Generate a plain-text list of 30 Yahoo Finance ticker symbols for US equities that meet these criteria:
    REQUIRED:
    - Listed on major exchanges in NYSE and NASDAQ
    - Mid-cap to large-cap (market cap greater than $500M USD)
    - Reasonable liquidity (average daily volume >$500k USD)
    - Growth potential: revenue growth >15% annually OR margin expansion trajectory
    - Value characteristics: P/E <50
    - NO PFIC reporting risks   
    FORMAT:
    - Yahoo Finance format tickers
    - One company per line with columns: Ticker, Company Name, Revenue Growth %, P/E Ratio, Market Cap
    - Market Cap is a number, the unit of which is millions of USD (e.g., 1500 = $1.5B)
    - Columns separated by commas. Column names have no leading or trailing spaces.
    - The first row should be the header row with these column titles: Ticker, Company Name, Revenue Growth %, P/E Ratio, Market Cap.
    - The rest of rows are sorted by revenue growth rate in descending order.
    - No explanations or additional text
    Focus on: industrials, technology, consumer discretionary, healthcare
    Exclude: financials, REITs, utilities, telecoms
    """
    response = model.generate_content(full_prompt)
    # Convert the CSV-formatted string to a pandas DataFrame
    df = pd.read_csv(StringIO(response.text))
    
    # Remove leading and trailing whitespaces from column names
    df.columns = df.columns.str.strip()

    return df


def plot_stock_analysis(stocks_df):

    # Create visualization: Revenue Growth vs P/E Ratio
    # Remove any rows with missing data
    
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
    print(f"Final Decision for {ticker.upper()} on {curr_date}: {decision}")
    return decision

list_of_stocks = [
('ARW', 'Arrow Electronics, Inc.'),
('AVT', 'Avnet, Inc.'),
('AMKR', 'Amkor Technology, Inc.'),
('VSH', 'Vishay Intertechnology, Inc.'),
('HPE', 'Hewlett Packard Enterprise Company'),
('DXC', 'DXC Technology Company'),
('SANM', 'Sanmina Corporation'),
('CNXC', 'Concentrix Corporation'),
('BWA', 'BorgWarner Inc.'),
('LKQ', 'LKQ Corporation'),
('LAD', 'Lithia & Driveway'),
('PAG', 'Penske Automotive Group, Inc.'),
('PVH', 'PVH Corp.'),
('URBN', 'Urban Outfitters, Inc.'),
('KBH', 'KB Home'),
('TOL', 'Toll Brothers, Inc.'),
('TMHC', 'Taylor Morrison Home Corporation'),
('PHM', 'PulteGroup, Inc.'),
('AGCO', 'AGCO Corporation'),
('OSK', 'Oshkosh Corporation'),
('HII', 'Huntington Ingalls Industries, Inc.'),
('TKR', 'The Timken Company'),
('OC', 'Owens Corning'),
('TXT', 'Textron Inc.'),
('ALK', 'Alaska Air Group, Inc.'),
('UAL', 'United Airlines Holdings, Inc.'),
('DAL', 'Delta Air Lines, Inc.'),
('CNC', 'Centene Corporation'),
('UTHR', 'United Therapeutics Corporation'),
('UHS', 'Universal Health Services, Inc.')
]
"""
   Ticker                Company Name  Revenue Growth %  P/E Ratio  Market Cap
0    SMCI   Super Micro Computer Inc.             143.0       25.4       28500
1     TDW              Tidewater Inc.              45.2       15.8        4800
2     APP        AppLovin Corporation              42.5       48.2       48000
3    POWL      Powell Industries Inc.              40.1       19.5        2100
4     NXT             Nextracker Inc.              35.4       28.1        7500
5     FIX    Comfort Systems USA Inc.              32.1       36.5       13200
6    FSLR            First Solar Inc.              27.3       18.2       22500
7     ANF     Abercrombie & Fitch Co.              22.1       18.4        8500
8    META         Meta Platforms Inc.              22.1       26.5     1250000
9    OSIS            OSI Systems Inc.              21.8       29.1        2300
10   DECK       Deckers Outdoor Corp.              20.3       29.4       23500
11   LNTH      Lantheus Holdings Inc.              20.1       19.2        7200
12   BRBR        BellRing Brands Inc.              19.5       34.1        7600
13   TTEK             Tetra Tech Inc.              19.2       39.5       13500
14    EME            EMCOR Group Inc.              18.4       26.8       18200
15    PWR        Quanta Services Inc.              17.5       42.1       40500
16   MEDP       Medpace Holdings Inc.              17.1       38.5       12100
17   BKNG       Booking Holdings Inc.              16.2       28.4      130000
18   ANET        Arista Networks Inc.              16.1       42.3       95000
19    HWM       Howmet Aerospace Inc.              15.4       39.2       33000
20  GOOGL               Alphabet Inc.              15.3       23.5     2100000
21    MOD    Modine Manufacturing Co.              15.1       24.2        6100
22    XYL                  Xylem Inc.              14.5       38.1       33500
23    THC      Tenet Healthcare Corp.              14.2       15.1       13800
24    VRT          Vertiv Holdings Co              13.5       41.2       35000
25   AMZN             Amazon.com Inc.              13.1       40.5     1900000
26     DY       Dycom Industries Inc.              10.5       25.4        5200
27    BLD              TopBuild Corp.               6.2       19.8       11500
28    PHM             PulteGroup Inc.               6.1        9.8       27000
29   GDDY                GoDaddy Inc.               5.9       19.5       19500
"""
"""
Best Opportunities (High Growth, Low P/E):
   Ticker                Company Name  Revenue Growth %  P/E Ratio  Market Cap
0    SMCI   Super Micro Computer Inc.             143.0       25.4       28500
1     TDW              Tidewater Inc.              45.2       15.8        4800
3    POWL      Powell Industries Inc.              40.1       19.5        2100
6    FSLR            First Solar Inc.              27.3       18.2       22500
7     ANF     Abercrombie & Fitch Co.              22.1       18.4        8500
8    META         Meta Platforms Inc.              22.1       26.5     1250000
11   LNTH      Lantheus Holdings Inc.              20.1       19.2        7200
14    EME            EMCOR Group Inc.              18.4       26.8       18200
"""

"""BUY: ANET, VRT, FSLR"""

if __name__ == "__main__":
    # Set up command line argument parser
    parser = argparse.ArgumentParser(description='Run trading analysis on a stock ticker')
    parser.add_argument('--ticker', '-t', type=str, default='',
                        help='Stock ticker symbol to analyze (default: NVDA)')
    parser.add_argument('--date', '-d', type=str, default=datetime.now().strftime('%Y-%m-%d'),
                        help='Analysis date in YYYY-MM-DD format (default: today)')
    parser.add_argument('--find-candidates', '-f', action='store_true',
                        help='Find stock candidates first using LLM before analysis')
    args = parser.parse_args()
    
    # If find_candidates flag is set, generate candidates and exit
    if args.find_candidates:
        stocks_df = find_stock_candidates()
        print("Generated Stock Candidates:")
        print(stocks_df)
        print(f"\nDataFrame shape: {stocks_df.shape}")
        print(f"Columns: {stocks_df.columns.tolist()}")
        find_best_stock_opportunities(stocks_df)
        plot_df = stocks_df.dropna(subset=['Revenue Growth %', 'P/E Ratio'])
        plot_stock_analysis(plot_df)
    else:    
        # Create a custom config
        config = DEFAULT_CONFIG.copy()
        config["llm_provider"] = "google"
        config["deep_think_llm"] = "gemini-3-pro-preview"  # Use a different model
        config["quick_think_llm"] = "gemini-2.5-flash"  # Use a different model
        config["max_debate_rounds"] = 1  # Increase debate rounds
        config["backend_url"] = ""

        # Configure data vendors (default uses yfinance and Alpha Vantage)
        config["data_vendors"] = {
            "core_stock_apis": "yfinance",           # Options: yfinance, alpha_vantage, local
            "technical_indicators": "yfinance",      # Options: yfinance, alpha_vantage, local
            "fundamental_data": "google",     # Options: openai, alpha_vantage, local
            "news_data": "google",            # Options: openai, alpha_vantage, google, local
        }

        # Initialize with custom config
        ta = TradingAgentsGraph(debug=True, config=config)
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