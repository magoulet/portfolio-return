from classes import Security, Cashflows
from pyxirr import xirr
from sqlalchemy import create_engine, text
from tabulate import tabulate
import argparse
import datetime as dt
import json
import logging
import logging.handlers as handlers
import os
import pandas as pd
import pathlib
import pickle
import psutil
import requests
import time
import yfinance as yf

from pymongo import MongoClient


# Read configuration
configuration_file = pathlib.Path('config.json')
config = json.loads(open(configuration_file).read())

# Parse arguments
parser = argparse.ArgumentParser(description='Program Flags')
parser.add_argument('-d', help='Snapshot date - format YYYY-MM-DD',
                    action="store", dest='date', default=False)
parser.add_argument('-m', help='Sends notification message to user',
                    action="store_true", dest='sendmail', default=False)
args = parser.parse_args()

# create a logger object instance
logger = logging.getLogger()

# specifies the lowest severity for logging
logger.setLevel(logging.NOTSET)

# set a destination for your logs or a "handler"
# here, we choose to print on console (a consoler handler)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.ERROR)

# set the logging format for your handler
log_format = '%(asctime)s | %(levelname)s: %(message)s'
console_handler.setFormatter(logging.Formatter(log_format))

# finally, we add the handler to the logger
logger.addHandler(console_handler)

# Create the rotating file handler. Limit the size to 100 KB.
# file_handler = logging.FileHandler('app.log')
file_handler = handlers.RotatingFileHandler('app.log', maxBytes=100000, backupCount=2)
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(logging.Formatter(log_format))
logger.addHandler(file_handler)

def read_cashflows(table, date, currency):
    cashflows = Cashflows(currency)
    df = read_db_contributions(table, date, currency)
    for index, row in df.iterrows():
        cashflows.event(row.date, row.contribution)

    return cashflows


def build_portfolio(table, date, currency):
    df = read_db_transactions(table, date, currency)

    # Adjust for stock splits
    df['Units'] = df['Units'] * df['SplitRatio']
    df['Price'] = df['Price'] / df['SplitRatio']

    tickers = df['Ticker'].unique().tolist()

    portfolio = {ticker: Security(ticker) for ticker in tickers}
    # Walk through the df
    df_dict = df.to_dict('records')
    for row in df_dict:
        if row['Type'].lower() == 'buy':
            portfolio[row['Ticker']].buy(row['Units'], row['Price'], row['Date'])
        elif row['Type'].lower() == 'sell':
            portfolio[row['Ticker']].sell(-1*row['Units'], row['Price'],
                                          row['Fees'], row['Date'])
    return portfolio


def get_price(tickers, date, path, offline=None):
    """
    Expected output format from yf:
                        Price
    Attributes Symbols
    Adj Close  BNS.TO   67.870003
               CAE.TO   33.270000
               ...      ...
    """
    prices = {}
    if not offline:
        logger.info(f"Getting prices for : {tickers}")
        df = yf.download(tickers, start=date-dt.timedelta(days=5), end=date+dt.timedelta(days=1),
                         group_by='Ticker')
        if df.empty:
            logger.error('Yahoo Finance returned an empty dataset!')
            exit()
        df.ffill(inplace=True) #fill missing values with most recent data
        result = df[-1:].stack(level=0, future_stack=True).rename_axis(['date', 'ticker']) \
            .reset_index(level=1)

        return result[['ticker', 'Adj Close']].set_index('ticker').to_dict()
    else:
        for ticker in tickers:
            with open(path+ticker+'.pickle', 'rb') as file:
                price = pickle.load(file).to_dict()
                prices[ticker] = price['Adj Close'][pd.Timestamp(date)]

        return {'Adj Close': prices}



def database_helper(sql_command, sql_operation):
    host = config["mysql"]["host"]
    user = config["mysql"]["user"]
    password = config["mysql"]["password"]
    database = config["mysql"]["database"]

    # Create the SQLAlchemy engine
    engine = create_engine(f"mysql+mysqlconnector://{user}:{password}@{host}/{database}")
    connection = engine.connect()

    if sql_operation == "Select":
        try:
            data = pd.read_sql(text(sql_command), connection)
        except Exception as e:
            logger.error("Cannot execute select from the database: ", str(e))
            data = None
    elif sql_operation == "Insert":
        trans = connection.begin()
        try:
            connection.execute(text(sql_command))
            trans.commit()
        except Exception as e:
            logger.error("Cannot insert into the database: ", str(e))
            trans.rollback()
        data = None
    else:
        logger.error("Invalid SQL operation:", sql_operation)
        data = None

    return data


def read_db_output(table):
    sql_command = "SELECT date, value FROM %s;" \
                  % table
    data = database_helper(sql_command, "Select")
    return data

def write_db_output(end, tot_value, tot_contributions, table):
    sql_command = "INSERT INTO %s VALUES ('%s', '%s', '%s') ON DUPLICATE KEY \
                 UPDATE value=VALUES(value), \
                 contributions=VALUES(contributions);" \
                 % (table, end, tot_value, tot_contributions)
    database_helper(sql_command, "Insert")
    return None


def read_db_transactions(table, date, currency):
    sql_command = "SELECT * FROM %s WHERE Date <= '%s' \
                  AND LOWER(Currency) = '%s';" \
                  % (table, date, currency)
    data = database_helper(sql_command, "Select")
    return data


def read_db_contributions(table, date, currency):
    sql_command = "SELECT * FROM %s WHERE Date <= '%s' \
                 AND LOWER(currency) = '%s';" \
                 % (table, date, currency)
    data = database_helper(sql_command, "Select")
    return data


def read_time_history(table, date):
    sql_command = "(SELECT * FROM %s WHERE date <= '%s' \
                  ORDER BY Date desc LIMIT 2) \
                  ORDER BY Date asc;" \
                  % (table, date)
    data = database_helper(sql_command, "Select")
    return data


def get_daily_variation(df):
    df['daily variation'] = df['value'].diff()
    daily_delta = df['daily variation'].iloc[-1]
    return daily_delta


def send_telegram_notification(body):

    url = 'https://api.telegram.org/bot{0}/{1}'.format(config['telegram']['token'],
                                                       config['telegram']['method'])
    params = {
        'chat_id': config['telegram']['chat_id'],
        'parse_mode': 'Markdown',
        'text': body
    }

    response = requests.post(url=url, params=params).json()

    return response


def store_results_in_mongo(results, currency, date):
    try:
        mongo_username = config['mongo']['username']
        mongo_password = config['mongo']['password']
        mongo_host = config['mongo']['host']
        mongo_port = config['mongo']['port']

        # Construct the MongoDB connection URI
        mongo_uri = f"mongodb://{mongo_username}:{mongo_password}@{mongo_host}:{mongo_port}/"

        # Connect to MongoDB
        client = MongoClient(mongo_uri)
        db = client['portfolio_tracker']
        collection = db[currency.lower()]

        # Convert date to a format that MongoDB can store
        # date_str = date.isoformat()  # Convert to a string in ISO format

        # Convert date to a datetime object
        date_datetime = dt.datetime.combine(date, dt.datetime.min.time())


        # Store the results as a document
        result_doc = {
            'date': date_datetime,
            'currency': currency,
            'total_value': results['total_value'],
            'total_contributions': results['total_contributions'],
            'total_unrealized_gain': results['total_unrealized_gain'],
            'total_realized_gain': results['total_realized_gain'],
            'mwrr': results['mwrr'],
            'portfolio_details': results['portfolio_details']
        }
        # collection.insert_one(result_doc)
        filter_criteria = {'date': date_datetime, 'currency': currency}
        collection.replace_one(filter_criteria, result_doc, upsert=True)
    except Exception as e:
        logger.error("Error storing results in MongoDB:", str(e))

def main():
    process = psutil.Process(os.getpid())
    exec_time = 0
    start_time = time.time()

    timing = eval(config["misc"]["Timing"])
    data_table = {'contributions': config["mysql"]["ContributionTable"],
                 'trades': config["mysql"]["TransactionTable"]}
    result_table = {'CAD': config["mysql"]["ResultTableCAD"],
                   'USD': config["mysql"]["ResultTableUSD"]}
    pickle_path = config["directories"]["pickles"]
    currencies = eval(config['misc']["Currencies"])

    if args.date:
        try:
            date = dt.datetime.strptime(args.date, "%Y-%m-%d").date()
        except ValueError:
            msg = "Not a valid date: '{0}'.".format(args.date)
            raise argparse.ArgumentTypeError(msg)
    else:
        date = dt.date.today()

    portfolio = {}

    for currency in currencies:
        portfolio[currency] = build_portfolio(data_table['trades'],
                                              date,
                                              currency,
                                              )

        # Get a list of active tickers (non-null qty) in the portfolio
        active_tickers = []
        for key, security in portfolio[currency].items():
            if security.qty > 0:
                active_tickers.append(security.ticker)

        prices = get_price(active_tickers, date, pickle_path)

        # Initialize variable that we'll use later
        tot_value = 0
        tot_money_in = 0
        tot_real_gain = 0
        raw_output_data = []

        # Iterate through all tickers in the portfolio, since inception
        # and execute calculations
        for key, security in portfolio[currency].items():
            if security.ticker in active_tickers:
                security.price = prices['Adj Close'][security.ticker]
                security.UnrealizedReturn = (security.price - security.costBasis) / \
                    security.costBasis * 100
                security.totUnrealizedReturn = security.qty * (security.price -
                                                         security.costBasis)
                tot_value += security.qty * security.price
                tot_money_in += security.moneyIn
                tot_real_gain += security.realGain
                output_row = [security.ticker,
                              security.qty,
                              security.price,
                              security.realGain,
                              security.qty*security.costBasis,
                              security.price*security.qty,
                              security.UnrealizedReturn,
                              security.totUnrealizedReturn]
                raw_output_data.append(output_row)
            else:
                tot_real_gain += security.realGain
        tot_unrealized_return = tot_value - tot_money_in
        # perc_unrealized_return = (tot_value - tot_money_in) / tot_money_in * 100

        # Prepare output
        output = pd.DataFrame(raw_output_data)
        output.columns = ['Ticker', 'Qty', 'Price', 'Real. Gain', 'Cost Basis',
                          'Value', 'Unreal. Gain {%}', 'Unreal. Gain ($)']

        output.sort_values('Unreal. Gain ($)', inplace=True)

        print(tabulate(output, headers="keys", floatfmt=".2f"))

        # Contributions to date
        cashflows = read_cashflows(
            data_table['contributions'], date, currency)
        tot_contributions = -1* cashflows.total() # IRR calculations: cash contributed is negative

        # Calculate MWRR (XIRR) since inception, not including today
        # Get all cashflows
        portfolio_values = read_db_output(result_table[currency])

        # Get initial and today's values
        initial_date, initial_value = portfolio_values.iloc[0]
        final_date = date
        final_value = tot_value

        # Add initial and today's values to the list
        cashflows.dates.extend([initial_date, final_date])
        cashflows.amounts.extend([-1*initial_value,final_value])

        # Create dataframe with dates and amounts
        df = pd.DataFrame({'dates': cashflows.dates, 'amount': cashflows.amounts})

        # Truncate the dataframe, then sort it
        df = df.loc[(df.dates >= initial_date) & (df.dates <= final_date) ]
        df.sort_values('dates', inplace=True)

        mwrr = xirr(df)

        print('\n*** Summary ({}): ***'.format(currency))
        print('Total Contributions: ${:,.0f}\n'
              'Total Value: ${:,.0f}\n'
              'Total Unrealized Gain: ${:,.0f}\n'
              'Total Realized Gain: ${:,.0f}\n'
              'Money Weighted Rate of Return (since {}): {:,.1f}%\n\n'
              .format(tot_contributions, tot_value, tot_unrealized_return,
                      tot_real_gain, initial_date.strftime("%Y-%m-%d"), mwrr*100))

        # Store the results in a dictionary
        results = {
            'total_value': tot_value,
            'total_contributions': tot_contributions,
            'total_unrealized_gain': tot_unrealized_return,
            'total_realized_gain': tot_real_gain,
            'mwrr': mwrr,
            'portfolio_details': output.to_dict('records')
        }


        logger.info("Writing to the DB...")
        # Store the results in MongoDB
        store_results_in_mongo(results, currency, date)

        write_db_output(date, round(tot_value, 2), tot_contributions,
                        result_table[currency])

        if args.sendmail:
            time_history = read_time_history(result_table[currency],
                                            dt.date.today())
            daily_delta = get_daily_variation(time_history)

            body = 'Daily variation ('+currency+'): $'\
                + str(daily_delta.round(2))\
                + '\nTotal value of the portfolio: $'\
                + str(round(tot_value, 2))
            logger.info(body)
            # sendmail(sender, to, subject, body)
            logger.info("Sending mail to user...")
            send_telegram_notification(body)

    finish_time = time.time()
    exec_time += finish_time-start_time
    logger.info('Total memory usage: {:,.0f} kb'.format(
        float(process.memory_info().rss)/1000))  # in bytes

    if timing:
        logger.info(f"Total Execution Time: {exec_time}")



if __name__ == "__main__":
    main()
