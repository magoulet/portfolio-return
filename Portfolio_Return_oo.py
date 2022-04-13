#!/usr/bin/env python3
#
# Notes:
# Filtering a DF: df[df['Type'] == 'Contr']

import argparse
import datetime as dt
import json
import mysql.connector
import numpy as np
import operator
import os
import psutil
import pandas as pd
import requests
import time
import yfinance as yf
from portfolio_classes import Security


def getConfigurations():
    path = os.path.dirname(os.path.abspath(__file__))
    configurationFile = path + '/config.json'
    configurations = json.loads(open(configurationFile).read())
    return configurations


def argParser():
    parser = argparse.ArgumentParser(description='Program Flags')
    parser.add_argument('-o', help="Output to Database (determined in config.json). Otherwise output to stdout only", action="store_true", dest='dboutput', default=False)
    parser.add_argument('-d', help='Snapshot date - format YYYY-MM-DD', action="store", dest='date', default=False)
    parser.add_argument('-m', help='Sends notification message to user', action="store_true", dest='sendmail', default=False)
    parser.add_argument('-rebalance', help='Rebalances Portfolio', action="store", dest='rebalance', default=False)
    parser.add_argument('-curr', help='Selected rebalancing currency', action="store", dest='currency', default='CAD')
    parser.add_argument('-broker', help='Selected rebalancing broker', action="store", dest='broker', default='dob')
    args = parser.parse_args()
    return args


def readContributions(SQLtable, date, currency):
    df = DBRead(configurations, SQLtable, date, currency)
    return df


def readTransactions(SQLtable, date, currency):
    portfolio = {}
    tickers = []

    df = DBRead(configurations, SQLtable, date, currency)

    # Adjust for stock splits
    df['Units'] = df['Units'] * df['SplitRatio']
    df['Price'] = df['Price'] / df['SplitRatio']


    tickers = df['Ticker'].unique().tolist()

    portfolio = {ticker: Security(ticker) for ticker in tickers}


    # Walk through the df
    for column, row in df.iterrows():
        if row['Type'].lower() == 'buy':
            portfolio[row['Ticker']].buy(row['Units'],row['Price'])
        elif row['Type'].lower() == 'sell':
            portfolio[row['Ticker']].sell(-1*row['Units'],row['Price'],row['Fees'])

    return portfolio


def get_price(tickers, date):
    '''
    Expected format:
                        Price
    Attributes Symbols
    Adj Close  BNS.TO   67.870003
               CAE.TO   33.270000
               ...      ...
    '''
    try:
        print("Getting prices for : ", tickers)
        print("Date: ", date)
        df = yf.download(tickers, start=date, end=date+dt.timedelta(days=1), group_by='Ticker')
        result = df[:1].stack(level=0).rename_axis(['date', 'ticker']).reset_index(level=1)

        return result[['ticker', 'Adj Close']].set_index('ticker').to_dict()
    except:
        print("Error getting online quotes...")
        exit()


def DatabaseHelper(sqlCommand, sqloperation, configurations):
    host = configurations["mysql"][0]["host"]
    user = configurations["mysql"][0]["user"]
    password = configurations["mysql"][0]["password"]
    database = configurations["mysql"][0]["database"]

    my_connect = mysql.connector.connect(host=host, user=user, password=password, database=database)
    cursor = my_connect.cursor()

    if sqloperation == "Select":
        try:
            data = pd.read_sql(sqlCommand, my_connect)
        except:
            print("Cannot select from the database...")
    if sqloperation == "Insert":
        try:
            cursor.execute(sqlCommand)
            my_connect.commit()
        except:
            my_connect.rollback()
        data = None

    my_connect.close()
    return data


def writeDBValue(end, TotValue, TotContributions, configurations, table):
    sqlCommand = "INSERT INTO %s VALUES ('%s', '%s', '%s') ON DUPLICATE KEY UPDATE value=VALUES(value), contributions=VALUES(contributions);" % (table.lower(), end, TotValue, TotContributions)
    DatabaseHelper(sqlCommand, "Insert", configurations)
    return None


def DBRead(configurations, table, date, currency):
    sqlCommand = "SELECT * FROM %s WHERE Date <= '%s'AND Currency = '%s' ORDER BY Date Asc;" % (table, date, currency)
    data = DatabaseHelper(sqlCommand, "Select", configurations)
    return data


def timeHistoryRead(configurations, table, date):
    sqlCommand = "SELECT * FROM %s WHERE date <= '%s' ORDER BY Date desc LIMIT 1;" % (table.lower(), date)
    data = DatabaseHelper(sqlCommand, "Select", configurations)
    return data


def get_daily_variation(df):
    df['daily variation'] = df['value'].diff()
    dailydelta = df['daily variation'].iloc[-1]
    return dailydelta




def rebalance(df, ExtraCash, broker, currency, end):
    print('\nRebalancing Portfolio with ${:,.0f} extra cash\n'.format(ExtraCash))
    # Current weight
    # Sort by AssetClass
    df = df[df['Broker'] == broker]
    Weight = df.pivot_table(index='AssetClass', values=['Value'], aggfunc=np.sum)
    TargetWeight = pd.read_csv('target_weight.csv')
    Weight = pd.merge(Weight, TargetWeight[TargetWeight['Currency'] == currency], on='AssetClass', how='inner')
    Weight['CurrWeight'] = Weight['Value'] / Weight['Value'].sum() * 100

    # Delta
    Weight['TargetPositions'] = Weight['TargetWeight'] * Weight['Value'].sum() / 100
    Weight['DeltaPositions'] = Weight['TargetPositions'] - Weight['Value']
    Weight['Extra'] = Weight['TargetWeight'] * ExtraCash / 100
    Weight['Delta'] = Weight['Extra'] + Weight['DeltaPositions']
    # ~ print(Weight)
    # ~ print(Weight.columns)
    # ~ input("debug 1...")

    # Merge the price column to the Weight dataframe
    Weight = pd.merge(Weight, df[['Ticker', 'Adj Close']], left_on='Ticker', right_on='Ticker', how='inner')

    Weight['NumUnits'] = (Weight['Delta'] / Weight['Adj Close']).round(4)

    print('\nPortfolio Rebalance:')
    print(Weight)


def telegramNotification(cfg, body):

    url = 'https://api.telegram.org/bot{0}/{1}'.format(cfg['token'],
                                                       cfg['method'])
    params = {
        'chat_id': cfg['chat_id'],
        'parse_mode': 'Markdown',
        'text': body
    }

    response = requests.post(url=url, params=params).json()

    return response


if __name__ == "__main__":
    process = psutil.Process(os.getpid())
    StartTime = time.time()

    configurations = getConfigurations()
    username = configurations["plotly"][0]["user"]
    api_key = configurations["plotly"][0]["api"]
    DownloadOnly = eval(configurations["misc"][0]["DownloadOnly"])
    OfflineData = eval(configurations["misc"][0]["OfflineData"])
    Timing = eval(configurations["misc"][0]["Timing"])
    TransactionTable = configurations["mysql"][0]["TransactionTable"]
    ContributionTable = configurations["mysql"][0]["ContributionTable"]
    ResultTableCAD = configurations["mysql"][0]["ResultTableCAD"]
    ResultTableUSD = configurations["mysql"][0]["ResultTableUSD"]

    args = argParser()

    if args.date:
        try:
            date = dt.datetime.strptime(args.date, "%Y-%m-%d").date()
        except ValueError:
            msg = "Not a valid date: '{0}'.".format(args.date)
            raise argparse.ArgumentTypeError(msg)
    else:
        date = dt.date.today()

    # if not marketopen(date):
    #     print("Markets closed on the chosen day!")
    #     quit()

    currencies = ['CAD', 'USD']

    for currency in currencies:
        portfolio = readTransactions(TransactionTable, date, currency)

        print("Getting online quotes...")
        tickers = []
        for key, value in portfolio.items():
            if value.qty > 0:
                tickers.append(value.ticker)
        prices = get_price(tickers, date)

        totValue = 0
        totMoneyIn = 0
        totRealGain = 0

        # for key, value in portfolio.items():
        for key, value in portfolio.items():
            if value.qty > 0:
                value.price = prices['Adj Close'][value.ticker]
                value.UnrealizedReturn = (value.price - value.costBasis)/ value.costBasis * 100
                value.totUnrealizedReturn = value.qty * (value.price - value.costBasis)
                totValue += value.qty * value.price
                totMoneyIn += value.moneyIn
                totRealGain += value.realGain
            else:
                totRealGain += value.realGain
        totUnrealizedReturn  = totValue - totMoneyIn
        percUnrealizedReturn = (totValue - totMoneyIn) / totMoneyIn * 100


        print ("{:<8} {:>8} {:>8} {:>12} {:>12} {:>8} {:>12} {:>12}".format('Ticker','Qty','Price','Real. Gain','Cost Basis','Value','Unreal. Gain {%}','Unreal. Gain ($)'))
        for v in sorted(portfolio.values(), key=operator.attrgetter('totUnrealizedReturn')):
            if v.qty > 0:
                print("{:<8} {:>8.2f} {:>8.2f} {:>12.2f} {:>12.2f} {:>8.2f} {:>12.2f} {:>12.2f}".format(v.ticker, v.qty, v.price, v.realGain, v.qty*v.costBasis, v.price*v.qty, v.UnrealizedReturn, v.totUnrealizedReturn))

        # Contributions to date
        Contributions = readContributions(ContributionTable, date, currency)
        TotContributions = Contributions[Contributions['currency'] == currency]['contribution'].sum()

        print('\n*** Summary ({}): ***'.format(currency))
        print('Total Contributions: ${:,.0f}\nTotal Value: ${:,.0f}\nTotal Unrealized Gain: ${:,.0f}\nTotal Realized Gain: ${:,.0f}\n\n'.format(TotContributions, totValue, totUnrealizedReturn, totRealGain))

        print("Writing to the DB...")
        writeDBValue(date, totValue, TotContributions, configurations,
                currency+'data')

        if args.sendmail:
            timeHistory = timeHistoryRead(configurations, currency+'data', dt.date.today())
            dailyDelta = get_daily_variation(timeHistory)

            body = 'Daily variation (currency): $' + str(dailyDelta.round(2)) + '\nTotal value of the portfolio: $' + str(totValue)

            print("Sending mail to user...")
            telegramNotification(config['telegram'][0], body)


    if args.rebalance:
        ExtraInvest = args.rebalance
        broker = args.broker
        rebalCurrency = args.currency
        rebalance(portfolio, float(ExtraInvest), broker, rebalCurrency, date)

    FinishTime = time.time()
    print('Total memory usage: {:,.0f} kb'.format(float(process.memory_info().rss)/1000))  # in bytes

    if Timing:
        print("Total Execution Time: ", FinishTime-StartTime)
