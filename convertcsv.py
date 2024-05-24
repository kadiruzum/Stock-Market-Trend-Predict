import yfinance as yf
import datetime
import pandas as pd

# List of stocks ending with ".IS"
stocks = ["AAPL","XU100.IS", "KCHOL.IS", "SASA.IS","EREGL.IS","FROTO.IS","BIMAS.IS","GUBRF.IS","BJKAS.IS"]  # Add all the relevant stocks

today = datetime.date.today()
duration = 3000
before = today - datetime.timedelta(days=duration)
start_date = before
end_date = today

# Create an empty list to store all the stock data
all_stock_data = []

for stock in stocks:
    df = yf.download(stock, start=start_date, end=end_date, progress=False)
    df['Stock'] = stock  # Add a column for the stock symbol
    all_stock_data.append(df)

# Concatenate all the dataframes into one
all_stock_data = pd.concat(all_stock_data)

# Save to CSV
all_stock_data.to_csv('dataset.csv')