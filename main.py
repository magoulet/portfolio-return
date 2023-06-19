from classes import Security, Cashflows
from pyxirr import xirr
from sqlalchemy import create_engine, text
from tabulate import tabulate
import argparse
import datetime as dt
import json
import os
import pandas as pd
import pickle
import psutil
import requests
import time
import yfinance as yf

def getconfig():
    path = os.path.dirname(os.path.abspath(__file__))
    configuration_file = path + '/config.json'
    config = json.loads(open(configuration_file).read())

    return config


def arg_parser():
    parser = argparse.ArgumentParser(description='Program Flags')
    parser.add_argument('-d', help='Snapshot date - format YYYY-MM-DD',
                        action="store", dest='date', default=False)
    parser.add_argument('-m', help='Sends notification message to user',
                        action="store_true", dest='sendmail', default=False)
    args = parser.parse_args()

    return args


def read_contributions(table, date, currency, config):
    cashflows = Cashflows(currency)
    df = read_db_contributions(config, table, date, currency)
    for index, row in df.iterrows():
        cashflows.event(row.date, row.contribution)

    return cashflows


def build_portfolio(table, date, currency, config):
    df = read_db_transactions(config, table, date, currency)

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
        print("Getting prices for : ", tickers)
        df = yf.download(tickers, start=date-dt.timedelta(days=5), end=date+dt.timedelta(days=1),
                         group_by='Ticker')
        if df.empty:
            print('DataFrame is empty!')
            exit()
        df.ffill(inplace=True) #fill missing values with most recent data
        result = df[-1:].stack(level=0).rename_axis(['date', 'ticker']) \
            .reset_index(level=1)

        return result[['ticker', 'Adj Close']].set_index('ticker').to_dict()
    else:
        for ticker in tickers:
            with open(path+ticker+'.pickle', 'rb') as file:
                price = pickle.load(file).to_dict()
                prices[ticker] = price['Adj Close'][pd.Timestamp(date)]

        return {'Adj Close': prices}



def database_helper(sql_command, sql_operation, config):
    host = config["mysql"][0]["host"]
    user = config["mysql"][0]["user"]
    password = config["mysql"][0]["password"]
    database = config["mysql"][0]["database"]

    # Create the SQLAlchemy engine
    engine = create_engine(f"mysql+mysqlconnector://{user}:{password}@{host}/{database}")
    connection = engine.connect()

    if sql_operation == "Select":
        try:
            data = pd.read_sql(text(sql_command), connection)
        except Exception as e:
            print("Cannot select from the database: ", str(e))
            data = None
    elif sql_operation == "Insert":
        trans = connection.begin()
        try:
            connection.execute(text(sql_command))
            trans.commit()
        except Exception as e:
            print("Cannot insert into the database: ", str(e))
            trans.rollback()
        data = None
    else:
        print("Invalid SQL operation:", sql_operation)
        data = None

    return data


def read_db_output(config, table):
    sql_command = "SELECT date, value FROM %s;" \
                  % table
    data = database_helper(sql_command, "Select", config)
    return data

def write_db_output(end, tot_value, tot_contributions, config, table):
    sql_command = "INSERT INTO %s VALUES ('%s', '%s', '%s') ON DUPLICATE KEY \
                 UPDATE value=VALUES(value), \
                 contributions=VALUES(contributions);" \
                 % (table, end, tot_value, tot_contributions)
    database_helper(sql_command, "Insert", config)
    return None


def read_db_transactions(config, table, date, currency):
    sql_command = "SELECT * FROM %s WHERE Date <= '%s' \
                  AND LOWER(Currency) = '%s';" \
                  % (table, date, currency)
    data = database_helper(sql_command, "Select", config)
    return data


def read_db_contributions(config, table, date, currency):
    sql_command = "SELECT * FROM %s WHERE Date <= '%s' \
                 AND LOWER(currency) = '%s';" \
                 % (table, date, currency)
    data = database_helper(sql_command, "Select", config)
    return data


def read_time_history(config, table, date):
    sql_command = "(SELECT * FROM %s WHERE date <= '%s' \
                  ORDER BY Date desc LIMIT 2) \
                  ORDER BY Date asc;" \
                  % (table, date)
    data = database_helper(sql_command, "Select", config)
    return data


def get_daily_variation(df):
    df['daily variation'] = df['value'].diff()
    daily_delta = df['daily variation'].iloc[-1]
    return daily_delta


def telegram_notification(cfg, body):

    url = 'https://api.telegram.org/bot{0}/{1}'.format(cfg['token'],
                                                       cfg['method'])
    params = {
        'chat_id': cfg['chat_id'],
        'parse_mode': 'Markdown',
        'text': body
    }

    response = requests.post(url=url, params=params).json()

    return response


def main():
    process = psutil.Process(os.getpid())
    exec_time = 0
    start_time = time.time()

    config = getconfig()
    timing = eval(config["misc"][0]["Timing"])
    data_table = {'contributions': config["mysql"][0]["ContributionTable"],
                 'trades': config["mysql"][0]["TransactionTable"]}
    result_table = {'CAD': config["mysql"][0]["ResultTableCAD"],
                   'USD': config["mysql"][0]["ResultTableUSD"]}
    path = config["directories"][0]["pickles"]
    currencies = eval(config['misc'][0]["Currencies"])

    args = arg_parser()

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
                                              config)

        # Get a list of active tickers (non-null qty) in the portfolio
        active_tickers = []
        for key, value in portfolio[currency].items():
            if value.qty > 0:
                active_tickers.append(value.ticker)

        prices = get_price(active_tickers, date, path)

        # Initialize variable that we'll use later
        tot_value = 0
        tot_money_in = 0
        tot_real_gain = 0
        raw_output_data = []

        # Iterate through all tickers in the portfolio, since inception
        # and execute calculations
        for key, value in portfolio[currency].items():
            if value.ticker in active_tickers:
                value.price = prices['Adj Close'][value.ticker]
                value.UnrealizedReturn = (value.price - value.costBasis) / \
                    value.costBasis * 100
                value.totUnrealizedReturn = value.qty * (value.price -
                                                         value.costBasis)
                tot_value += value.qty * value.price
                tot_money_in += value.moneyIn
                tot_real_gain += value.realGain
                output_row = [value.ticker,
                              value.qty,
                              value.price,
                              value.realGain,
                              value.qty*value.costBasis,
                              value.price*value.qty,
                              value.UnrealizedReturn,
                              value.totUnrealizedReturn]
                raw_output_data.append(output_row)
            else:
                tot_real_gain += value.realGain
        tot_unrealized_return = tot_value - tot_money_in
        # perc_unrealized_return = (tot_value - tot_money_in) / tot_money_in * 100

        # Prepare output
        output = pd.DataFrame(raw_output_data)
        output.columns = ['Ticker', 'Qty', 'Price', 'Real. Gain', 'Cost Basis',
                          'Value', 'Unreal. Gain {%}', 'Unreal. Gain ($)']

        output.sort_values('Unreal. Gain ($)', inplace=True)

        print(tabulate(output, headers="keys", floatfmt=".2f"))

        # Contributions to date
        cashflows = read_contributions(
            data_table['contributions'], date, currency, config)
        tot_contributions = -1* cashflows.total() # IRR calculations: cash contributed is negative

        # Calculate MWRR (XIRR) since inception, not including today
        # Get all cashflows
        portfolio_values = read_db_output(config, result_table[currency])

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

        print("Writing to the DB...")
        write_db_output(date, round(tot_value, 2), tot_contributions, config,
                        result_table[currency])

        if args.sendmail:
            time_history = read_time_history(config, result_table[currency],
                                            dt.date.today())
            daily_delta = get_daily_variation(time_history)

            body = 'Daily variation ('+currency+'): $'\
                + str(daily_delta.round(2))\
                + '\nTotal value of the portfolio: $'\
                + str(round(tot_value, 2))
            print(body)
            # sendmail(sender, to, subject, body)
            print("Sending mail to user...")
            telegram_notification(config['telegram'][0], body)

    finish_time = time.time()
    exec_time += finish_time-start_time
    print('Total memory usage: {:,.0f} kb'.format(
        float(process.memory_info().rss)/1000))  # in bytes

    if timing:
        print("Total Execution Time: ", exec_time)



if __name__ == "__main__":
    main()
