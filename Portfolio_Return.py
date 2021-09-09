#!/usr/bin/env python3
#
# Notes:
# Filtering a DF: df[df['Type'] == 'Contr']

import argparse
from botMsg import telegram_bot_sendtext as sendtext
import datetime as dt
import json
import mysql.connector
import numpy as np
import os
import pandas as pd
import time
import yfinance as yf


def getconfig():
    path = os.path.dirname(os.path.abspath(__file__))
    configurationFile = path + '/config.json'
    config = json.loads(open(configurationFile).read())

    return config


def argParser():
    parser = argparse.ArgumentParser(description='Program Flags')
    parser.add_argument('-d', help='Snapshot date - format YYYY-MM-DD', action="store", dest='date', default=False)
    parser.add_argument('-m', help='Sends notification message to user', action="store_true", dest='sendmail', default=False)
    parser.add_argument('-rebalance', help='Rebalances Portfolio', action="store", dest='rebalance', default=False)
    parser.add_argument('-curr', help='Selected rebalancing currency', action="store", dest='currency', default='CAD')
    parser.add_argument('-broker', help='Selected rebalancing broker', action="store", dest='broker', default='dob')
    args = parser.parse_args()

    return args


def readContributions(table, date, currency):
    df = readDBContributions(config, table, date, currency)
    return df


def readTransactions(table, date, currency):
    df = readDBTransactions(config, table, date, currency)

    df.sort_values(by=['Date'], ascending=True, inplace=True)

    # Adjust for stock splits
    df['Units'] = df['Units'] * df['SplitRatio']
    df['Price'] = df['Price'] / df['SplitRatio']

    # Add average cost basis
    df['PurchasePrice'] = df[['Units', 'Price']].apply(lambda x: x['Units']*x['Price'] if x['Units'] > 0 else 0, axis=1)
    df['Proceeds'] = df[['Units', 'Price']].apply(lambda x: -1*x['Units']*x['Price'] if x['Units'] < 0 else 0, axis=1)
    df['totUnits'] = df.groupby(['Ticker'])['Units'].cumsum()

    # Walk through df in historical order
    df = df.groupby('Ticker')
    result = pd.DataFrame()

    for name, group in df:
        df2 = df.get_group(name).reset_index(drop=True)

        df2.loc[0, 'totPurchaseCost'] = df2.loc[0, 'PurchasePrice']
        df2.loc[0, 'AvgCost'] = df2.loc[0, 'totPurchaseCost'] / df2.loc[0, 'totUnits']

        for index, row in df2.iterrows():
            if index > 0:
                # Buy order
                if row['Units'] > 0:
                    df2.loc[index, 'totPurchaseCost'] = df2.loc[index, 'PurchasePrice'] + df2.loc[index-1, 'totPurchaseCost']
                    df2.loc[index, 'AvgCost'] = df2.loc[index, 'totPurchaseCost'] / df2.loc[index, 'totUnits']
                    df2.loc[index, 'CostBasis'] = np.nan
                    df2.loc[index, 'RealGain'] = np.nan

                # Sell order
                else:
                    df2.loc[index, 'AvgCost'] = df2.loc[index-1, 'AvgCost']
                    df2.loc[index, 'totPurchaseCost'] = df2.loc[index, 'totUnits'] * df2.loc[index, 'AvgCost']
                    df2.loc[index, 'CostBasis'] = -1 * df2.loc[index, 'AvgCost'] * df2.loc[index, 'Units']
                    df2.loc[index, 'RealGain'] = df2.loc[index, 'Proceeds'] - df2.loc[index, 'CostBasis']

        result = result.append(df2, ignore_index=True, sort=True)
    return result


def getPrice(tickers, date):
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
        df = yf.download(tickers, start=date, end=date+dt.timedelta(days=1),
                        group_by='Ticker')
        if df.empty:
            print('DataFrame is empty!')
            exit()
        result = df[:1].stack(level=0).rename_axis(['date', 'ticker']) \
            .reset_index(level=1)

        return result[['ticker', 'Adj Close']].set_index('ticker')
    except:
        print("Error getting online quotes...")
        exit()


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
    Weight = df.pivot_table(index='AssetClass', values=['Value'], \
        aggfunc=np.sum)
    TargetWeight = pd.read_csv('target_weight.csv')
    Weight = pd.merge(Weight, TargetWeight[TargetWeight['Currency'] == \
        currency], on='AssetClass', how='inner')
    Weight['CurrWeight'] = Weight['Value'] / Weight['Value'].sum() * 100

    # Delta
    Weight['TargetPositions'] = Weight['TargetWeight'] * \
        Weight['Value'].sum() / 100
    Weight['DeltaPositions'] = Weight['TargetPositions'] - Weight['Value']
    Weight['Extra'] = Weight['TargetWeight'] * ExtraCash / 100
    Weight['Delta'] = Weight['Extra'] + Weight['DeltaPositions']

    # Merge the price column to the Weight dataframe
    Weight = pd.merge(Weight, df[['Ticker', 'Adj Close']], left_on='Ticker', \
        right_on='Ticker', how='inner')

    Weight['NumUnits'] = (Weight['Delta'] / Weight['Adj Close']).round(4)

    print('\nPortfolio Rebalance:')
    print(Weight)


if __name__ == "__main__":
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
    currencies = ['CAD','USD']

    for currency in currencies:
        df = readTransactions(dataTable['trades'], date, currency)
        portfolio[currency] = df.pivot_table(index=['Ticker', 'Broker', 'AssetClass', 'Currency'], values=['totUnits', 'totPurchaseCost', 'RealGain'], aggfunc={'totUnits': 'last', 'totPurchaseCost': 'last', 'RealGain': 'sum'}, fill_value=0)

        portfolio[currency].reset_index(inplace=True)
        TotRealGain = portfolio[currency]['RealGain'].sum().round(2)

        # Remove tickers with 0 units
        portfolio[currency] = portfolio[currency][portfolio[currency]['totUnits'] != 0]

        print("Getting online quotes...")
        tickers = portfolio[currency]['Ticker'].to_list()
        prices = getPrice(tickers, date)
        portfolio[currency] = pd.merge(portfolio[currency], prices, left_on='Ticker', right_on='ticker', how='inner')

        # Portfolio Value
        portfolio[currency]['Value'] = portfolio[currency]['Adj Close'] * portfolio[currency]['totUnits']
        TotValue = portfolio[currency]['Value'].sum().round(2)

        # Unrealized Gain/Loss
        portfolio[currency]['unrealizedGainPerc'] = (portfolio[currency]['Value'] - portfolio[currency]['totPurchaseCost']) / portfolio[currency]['totPurchaseCost'] * 100
        portfolio[currency]['totUnrealizedGain'] = portfolio[currency]['Value'] - portfolio[currency]['totPurchaseCost']
        portfolio[currency].sort_values(by=['totUnrealizedGain'], inplace=True)
        TotUnrealGain = portfolio[currency]['totUnrealizedGain'].sum().round(2)

        # Contributions to date
        contributions['currency'] = readContributions(dataTable['contributions'], date, currency)
        TotContributions = contributions['currency']['contribution'].sum()

        print('\n*** Summary ({}): ***'.format(currency))
        print('Total Contributions: ${:,.0f}\n'
                'Total Value: ${:,.0f}\n'
                'Total Unrealized Gain: ${:,.0f}\n'
                'Total Realized Gain: ${:,.0f}\n\n'
                .format(TotContributions, TotValue, TotUnrealGain, TotRealGain))
        print(portfolio[currency].round(2))

        print("Writing to the DB...")
        writeDBValue(date, TotValue, TotContributions, config,
                resultTable[currency])

        if args.sendmail:
            TimeHistory = timeHistoryRead(config, resultTable[currency], dt.date.today())
            dailydelta = getDailyVariation(TimeHistory)

            body = 'Daily variation ('+currency+'): $' + str(dailydelta.round(2)) + '\nTotal value of the portfolio: $' + str(TotValue)

            # sendmail(sender, to, subject, body)
            print("Sending mail to user...")
            sendtext(body)


    if args.rebalance:
        ExtraInvest = args.rebalance
        broker = args.broker
        rebalCurrency = args.currency
        rebalance(portfolio[rebalCurrency], float(ExtraInvest), broker, rebalCurrency, date)

    FinishTime = time.time()
    if Timing:
        print("Total Execution Time: ", FinishTime-StartTime)
