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
import pandas as pd
import pickle
import psutil
import requests
import time
import yfinance as yf
from portfolio_classes import Security


def getconfig():
    path = os.path.dirname(os.path.abspath(__file__))
    configurationFile = path + '/config.json'
    config = json.loads(open(configurationFile).read())

    return config


def argParser():
    parser = argparse.ArgumentParser(description='Program Flags')
    parser.add_argument('-d', help='Snapshot date - format YYYY-MM-DD',
                        action="store", dest='date', default=False)
    parser.add_argument('-m', help='Sends notification message to user',
                        action="store_true", dest='sendmail', default=False)
    parser.add_argument('-rebalance', help='Rebalances Portfolio',
                        action="store", dest='rebalance', default=False)
    parser.add_argument('-curr', help='Selected rebalancing currency',
                        action="store", dest='currency', default='CAD')
    parser.add_argument('-broker', help='Selected rebalancing broker',
                        action="store", dest='broker', default='dob')
    args = parser.parse_args()

    return args


def readContributions(table, date, currency):
    df = readDBContributions(config, table, date, currency)
    return df


def readTransactions(table, date, currency, config):
    df = readDBTransactions(config, table, date, currency)
    portfolio = {}
    tickers = []

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


def getPrice(tickers, date, path):
    '''
    Expected output format from yf:
                        Price
    Attributes Symbols
    Adj Close  BNS.TO   67.870003
               CAE.TO   33.270000
               ...      ...
    '''
    prices = {}
    try:
        for ticker in tickers:
            with open(path+ticker+'.pickle', 'rb') as file:
                price = pickle.load(file).to_dict()
                prices[ticker] = price['Adj Close'][pd.Timestamp(date)]

        return {'Adj Close': prices}

    except Exception:
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



def databaseHelper(sqlCommand, sqloperation, config):
    host = config["mysql"][0]["host"]
    user = config["mysql"][0]["user"]
    password = config["mysql"][0]["password"]
    database = config["mysql"][0]["database"]

    my_connect = mysql.connector.connect(host=host, user=user,
                                         password=password,
                                         database=database)
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


def writeDBValue(end, TotValue, TotContributions, config, table):
    sqlCommand = "INSERT INTO %s VALUES ('%s', '%s', '%s') ON DUPLICATE KEY \
                 UPDATE value=VALUES(value), \
                 contributions=VALUES(contributions);" \
                 % (table, end, TotValue, TotContributions)
    databaseHelper(sqlCommand, "Insert", config)
    return None


def readDBTransactions(config, table, date, currency):
    sqlCommand = "SELECT * FROM %s WHERE Date <= '%s' \
                  AND LOWER(Currency) = '%s';" \
                  % (table, date, currency)
    data = databaseHelper(sqlCommand, "Select", config)
    return data


def readDBContributions(config, table, date, currency):
    sqlCommand = "SELECT * FROM %s WHERE Date <= '%s' \
                 AND LOWER(currency) = '%s';" \
                 % (table, date, currency)
    data = databaseHelper(sqlCommand, "Select", config)
    return data


def timeHistoryRead(config, table, date):
    sqlCommand = "(SELECT * FROM %s WHERE date <= '%s' \
                  ORDER BY Date desc LIMIT 2) \
                  ORDER BY Date asc;" \
                  % (table, date)
    data = databaseHelper(sqlCommand, "Select", config)
    return data


def getDailyVariation(df):
    df['daily variation'] = df['value'].diff()
    dailydelta = df['daily variation'].iloc[-1]
    return dailydelta


def rebalance(df, ExtraCash, broker, currency, end):
    print('\nRebalancing Portfolio with ${:,.0f} extra cash\n'
          .format(ExtraCash))
    # Current weight
    # Sort by AssetClass
    df = df[df['Broker'] == broker]
    Weight = df.pivot_table(index='AssetClass', values=['Value'],
                            aggfunc=np.sum)
    TargetWeight = pd.read_csv('target_weight.csv')
    Weight = pd.merge(Weight, TargetWeight[TargetWeight['Currency'] ==
                      currency], on='AssetClass', how='inner')
    Weight['CurrWeight'] = Weight['Value'] / Weight['Value'].sum() * 100

    # Delta
    Weight['TargetPositions'] = Weight['TargetWeight'] * \
        Weight['Value'].sum() / 100
    Weight['DeltaPositions'] = Weight['TargetPositions'] - Weight['Value']
    Weight['Extra'] = Weight['TargetWeight'] * ExtraCash / 100
    Weight['Delta'] = Weight['Extra'] + Weight['DeltaPositions']

    # Merge the price column to the Weight dataframe
    Weight = pd.merge(Weight, df[['Ticker', 'Adj Close']], left_on='Ticker',
                      right_on='Ticker', how='inner')

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
    execTime = 0
    StartTime = time.time()

    config = getconfig()
    username = config["plotly"][0]["user"]
    api_key = config["plotly"][0]["api"]
    DownloadOnly = eval(config["misc"][0]["DownloadOnly"])
    OfflineData = eval(config["misc"][0]["OfflineData"])
    Timing = eval(config["misc"][0]["Timing"])
    dataTable = {'contributions': config["mysql"][0]["ContributionTable"],
                 'trades': config["mysql"][0]["TransactionTable"]}
    resultTable = {'CAD': config["mysql"][0]["ResultTableCAD"],
                   'USD': config["mysql"][0]["ResultTableUSD"]}
    path = config["directories"][0]["pickles"]
    currencies = eval(config['misc'][0]["Currencies"])

    args = argParser()

    if args.date:
        try:
            date = dt.datetime.strptime(args.date, "%Y-%m-%d").date()
        except ValueError:
            msg = "Not a valid date: '{0}'.".format(args.date)
            raise argparse.ArgumentTypeError(msg)
    else:
        date = dt.date.today()

    portfolio = {}
    contributions = {}

    for currency in currencies:
        portfolio[currency] = readTransactions(dataTable['trades'], date,
                                               currency, config)

        tickers = []
        for key, value in portfolio[currency].items():
            if value.qty > 0:
                tickers.append(value.ticker)

        prices = getPrice(tickers, date, path)

        totValue = 0
        totMoneyIn = 0
        totRealGain = 0

        # for key, value in portfolio[currency].items():
        for key, value in portfolio[currency].items():
            if value.qty > 0:
                value.price = prices['Adj Close'][value.ticker]
                value.UnrealizedReturn = (value.price - value.costBasis) / \
                    value.costBasis * 100
                value.totUnrealizedReturn = value.qty * (value.price -
                                                         value.costBasis)
                totValue += value.qty * value.price
                totMoneyIn += value.moneyIn
                totRealGain += value.realGain
            else:
                totRealGain += value.realGain
        totUnrealizedReturn = totValue - totMoneyIn
        percUnrealizedReturn = (totValue - totMoneyIn) / totMoneyIn * 100

        print("{:<8} {:>8} {:>8} {:>12} {:>12} {:>9} {:>12} {:>12}".
              format('Ticker', 'Qty', 'Price', 'Real. Gain', 'Cost Basis',
                     'Value', 'Unreal. Gain {%}', 'Unreal. Gain ($)'))
        for v in sorted(portfolio[currency].values(), key=operator.
                        attrgetter('totUnrealizedReturn')):
            if v.qty > 0:
                print(('{:<8} {:>8.2f} {:>8.2f} {:>12.2f} {:>12.2f} {:>9.2f}'
                      '{:>17.1f} {:>16.2f}').format(v.ticker, v.qty, v.price,
                                                    v.realGain,
                                                    v.qty*v.costBasis,
                                                    v.price*v.qty,
                                                    v.UnrealizedReturn,
                                                    v.totUnrealizedReturn))

        # Contributions to date
        contributions['currency'] = readContributions(
            dataTable['contributions'], date, currency)
        TotContributions = contributions['currency']['contribution'].sum()

        print('\n*** Summary ({}): ***'.format(currency))
        print('Total Contributions: ${:,.0f}\n'
              'Total Value: ${:,.0f}\n'
              'Total Unrealized Gain: ${:,.0f}\n'
              'Total Realized Gain: ${:,.0f}\n\n'
              .format(TotContributions, totValue, totUnrealizedReturn,
                      totRealGain))

        print("Writing to the DB...")
        writeDBValue(date, round(totValue, 2), TotContributions, config,
                     resultTable[currency])

        if args.sendmail:
            TimeHistory = timeHistoryRead(config, resultTable[currency],
                                          dt.date.today())
            dailydelta = getDailyVariation(TimeHistory)

            body = 'Daily variation ('+currency+'): $'\
                + str(dailydelta.round(2))\
                + '\nTotal value of the portfolio: $'\
                + str(round(totValue, 2))
            print(body)
            # sendmail(sender, to, subject, body)
            print("Sending mail to user...")
            telegramNotification(config['telegram'][0], body)

    if args.rebalance:
        ExtraInvest = args.rebalance
        broker = args.broker
        rebalCurrency = args.currency
        rebalance(portfolio[rebalCurrency], float(ExtraInvest), broker,
                  rebalCurrency, date)

    FinishTime = time.time()
    execTime += FinishTime-StartTime
    print('Total memory usage: {:,.0f} kb'.format(
        float(process.memory_info().rss)/1000))  # in bytes

    if Timing:
        print("Total Execution Time: ", execTime)
