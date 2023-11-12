# Portfolio Return Calculator

This is a Portfolio Return Calculator that takes a list of transactions over time as input and computes various calculations related to the portfolio's return.

## Table of Contents
- [Introduction](#introduction)
- [Requirements](#requirements)
- [Usage](#usage)
- [Configuration](#configuration)
- [Logging](#logging)
- [Data Retrieval](#data-retrieval)
- [Database Operations](#database-operations)
- [Portfolio Calculations](#portfolio-calculations)
- [Output](#output)
- [Notification](#notification)
- [Execution Time](#execution-time)
- [Conclusion](#conclusion)

## Introduction

The Portfolio Return Calculator is designed to analyze a portfolio's performance based on a series of transactions. It calculates various metrics such as the total value, unrealized gains, realized gains, and the money-weighted rate of return (MWR). The calculations are performed for multiple currencies and stored in a database for future reference.

## Requirements

The following libraries are required to run the Portfolio Return Calculator:
- `classes` - Custom classes for securities and cashflows.
- `pyxirr` - Library for calculating the money-weighted rate of return (MWR).
- `sqlalchemy` - Library for connecting to the MySQL database.
- `tabulate` - Library for formatting the output table.
- `argparse` - Library for parsing command-line arguments.
- `datetime` - Library for working with dates and times.
- `json` - Library for working with JSON configuration files.
- `logging` - Library for logging messages to a file.
- `logging.handlers` - Library for handling log file rotation.
- `os` - Library for interacting with the operating system.
- `pandas` - Library for data manipulation and analysis.
- `pathlib` - Library for working with file paths.
- `pickle` - Library for object serialization.
- `psutil` - Library for retrieving system information.
- `requests` - Library for making HTTP requests.
- `time` - Library for working with time intervals.
- `yfinance` - Library for retrieving financial data from Yahoo Finance.

Make sure to install these libraries before using the Portfolio Return Calculator.

## Usage

To use the Portfolio Return Calculator, you need to provide a list of transactions as input. The program reads the transactions from a database table and performs the necessary calculations. The transactions should include information such as the ticker symbol, date, quantity, price, fees, and transaction type (buy/sell).

The program can be executed with the following command-line arguments:
- `-d` or `--date`: Specify a snapshot date in the format YYYY-MM-DD.
- `-m` or `--sendmail`: Enable email or Telegram notification to the user.

Run the program using the following command:
```shell
python main.py [-d SNAPSHOT_DATE] [-m]
```

Replace `SNAPSHOT_DATE` with the desired snapshot date in the format mentioned above.

## Configuration

The configuration of the Portfolio Return Calculator is stored in a JSON file named `config.json`. This file contains various settings such as database credentials, table names, directories, and miscellaneous options.

To configure the program, modify the `config.json` file according to your requirements.

## Logging

The Portfolio Return Calculator logs messages to help track the program's execution and troubleshoot any issues. It uses the `logging` module to create a logger object and set up log handlers. The logs are stored in a file named `app.log`, which is rotated when it exceeds a certain size.

The log level is set to `logging.ERROR`, so only error-level messages are displayed on the console. You can modify the log level and format by adjusting the code in the `main` function.

## Data Retrieval

The program retrieves data from various sources to perform the portfolio calculations. It reads the configuration file to determine the location of the data and the APIs to use.

- Reading from a database: The program uses the `database_helper` function to execute SQL commands and retrieve data from a MySQL database. It connects to the database using the credentials specified in the configuration file.

- Retrieving stock prices: The program uses the `yfinance` library to download historical stock prices from Yahoo Finance. It retrieves the prices for a list of tickers and a specified date range. If offline mode is enabled, it reads pre-downloaded pickle files containing the stock prices.

## Database Operations

The Portfolio Return Calculator performs various database operations to store and retrieve data. It uses the `sqlalchemy` library to establish a connection with the MySQL database.

The program provides helper functions to execute SQL commands and retrieve data from the database. It supports SELECT and INSERT operations and handles exceptions gracefully.

## Portfolio Calculations

The heart of the Portfolio Return Calculator lies in the calculations performed on the portfolio. It builds a portfolio based on the transactions retrieved from the database. The portfolio is a collection of securities, each represented by a ticker symbol.

The program calculates various metrics and values for each security in the portfolio, such as unrealized return, total unrealized return, value, cost basis, and realized gain. These calculations are then aggregated to calculate the total value, total unrealized gain, total realized gain, and money-weighted rate of return (MWR).

## Output

The program outputs the results of the portfolio calculations in a tabular format. The `tabulate` library is used to generate the table, which includes information such as tickers, quantities, prices, realized gains, cost basis, values, unrealized gains (%), and unrealized gains ($).

The output also includes a summary section that provides the total contributions, total value, total unrealized gain, total realized gain, and the money-weighted rate of return (MWR) since the portfolio's inception.

## Notification

The Portfolio Return Calculator can send a notification message to the user via Telegram. The program utilizes the Telegram Bot API to send notifications. To enable this feature, set the `sendmail` flag to `True` in the command-line arguments, and provide the necessary API credentials in the configuration file.

The notification includes the daily variation of the portfolio's value and the total value of the portfolio. Adjust the notification method according to your needs by modifying the `send_telegram_notification` function.

