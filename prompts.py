STOCK_LIST = """Generate a plain-text list of 30 Yahoo Finance ticker symbols for US equities that meet these criteria:
    REQUIRED:
    - Listed on major exchanges in NYSE and NASDAQ
    - Mid-cap to large-cap (market cap greater than $500M USD)
    - Reasonable liquidity (average daily volume >$500k USD)
    - Growth potential: revenue growth >15% in the latest earning report OR margin expansion trajectory.
    - Value characteristics: P/E <50
    - RoE is between 10% and 40%
    - Positive cash flow from operations in the latest quarterly report
    - Low debt-to-equity ratio (<1.0)
    - NO PFIC reporting risks   
    - Please use financial data from the last 3 months. If necessary, you can search the web to get the latest financial data.
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
