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


def getConfigurations():
    path = os.path.dirname(os.path.abspath(__file__))
    configurationFile = path + '/config.json'
    configurations = json.loads(open(configurationFile).read())

    return configurations


def ArgParser():
    parser = argparse.ArgumentParser(description='Program Flags')
    parser.add_argument('-o', help="Output to Database (determined in config.json). Otherwise output to stdout only", action="store_true", dest='dboutput', default=False)
    parser.add_argument('-d', help='Snapshot date - format YYYY-MM-DD', action="store", dest='date', default=False)
    parser.add_argument('-m', help='Sends notification message to user', action="store_true", dest='sendmail', default=False)
    parser.add_argument('-rebalance', help='Rebalances Portfolio', action="store", dest='rebalance', default=False)
    parser.add_argument('-curr', help='Selected rebalancing currency', action="store", dest='currency', default='CAD')
    parser.add_argument('-broker', help='Selected rebalancing broker', action="store", dest='broker', default='dob')
    args = parser.parse_args()

    return args


def readContributions(table, date):
    df = DBRead(configurations, table, date)
    return df


def readTransactions(table, date):
    df = DBRead(configurations, table, date)

    df.sort_values(by=['Date'], ascending=True, inplace=True)

    # Adjust for stock splits
    df['Units'] = df['Units'] * df['SplitRatio']
    df['Price'] = df['Price'] / df['SplitRatio']

    # Add average cost basis
    df['PurchasePrice'] = df[['Units', 'Price']].apply(lambda x: x['Units']*x['Price'] if x['Units'] > 0 else 0, axis=1)
    df['Proceeds'] = df[['Units', 'Price']].apply(lambda x: -1*x['Units']*x['Price'] if x['Units'] < 0 else 0, axis=1)
    df['CumulUnits'] = df.groupby(['Ticker'])['Units'].cumsum()

    # Walk through df in historical order
    df = df.groupby('Ticker')
    result = pd.DataFrame()

    for name, group in df:
        df2 = df.get_group(name).reset_index(drop=True)

        df2.loc[0, 'SumPurchaseCost'] = df2.loc[0, 'PurchasePrice']
        df2.loc[0, 'AvgCost'] = df2.loc[0, 'SumPurchaseCost'] / df2.loc[0, 'CumulUnits']

        for index, row in df2.iterrows():
            if index > 0:
                # Buy order
                if row['Units'] > 0:
                    df2.loc[index, 'SumPurchaseCost'] = df2.loc[index, 'PurchasePrice'] + df2.loc[index-1, 'SumPurchaseCost']
                    df2.loc[index, 'AvgCost'] = df2.loc[index, 'SumPurchaseCost'] / df2.loc[index, 'CumulUnits']
                    df2.loc[index, 'CostBasis'] = np.nan
                    df2.loc[index, 'RealGain'] = np.nan

                # Sell order
                else:
                    df2.loc[index, 'AvgCost'] = df2.loc[index-1, 'AvgCost']
                    df2.loc[index, 'SumPurchaseCost'] = df2.loc[index, 'CumulUnits'] * df2.loc[index, 'AvgCost']
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
        print("Date: ", date)
        df = yf.download(tickers, start=date, end=date+dt.timedelta(days=1), group_by='Ticker')
        if df.empty:
            print('DataFrame is empty!')
            exit()
        result = df[:1].stack(level=0).rename_axis(['date', 'ticker']).reset_index(level=1)

        return result[['ticker', 'Adj Close']].set_index('ticker')
    except:
        print("Error getting online quotes...")
        exit()


def databaseHelper(sqlCommand, sqloperation, configurations):
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
    sqlCommand = "INSERT INTO %s VALUES ('%s', '%s', '%s') ON DUPLICATE KEY UPDATE value=VALUES(value), contributions=VALUES(contributions);" % (table, end, TotValue, TotContributions)
    databaseHelper(sqlCommand, "Insert", configurations)
    return None


def DBRead(configurations, table, date):
    sqlCommand = "SELECT * FROM %s WHERE Date <= '%s';" % (table, date)
    data = databaseHelper(sqlCommand, "Select", configurations)
    return data

def timeHistoryRead(configurations, table, date):
    sqlCommand = "SELECT * FROM %s WHERE date <= '%s' ORDER BY Date desc LIMIT 1;" % (table, date)
    data = databaseHelper(sqlCommand, "Select", configurations)
    return data

def getDailyVariation(df):
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


if __name__ == "__main__":
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

    args = ArgParser()

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


    df = readTransactions(TransactionTable, date)
    portfolio = df.pivot_table(index=['Ticker', 'Broker', 'AssetClass', 'Currency'], values=['CumulUnits', 'SumPurchaseCost', 'RealGain'], aggfunc={'CumulUnits': 'last', 'SumPurchaseCost': 'last', 'RealGain': 'sum'}, fill_value=0)

    portfolio.reset_index(inplace=True)

    TotRealGainCAD = portfolio[portfolio['Currency'] == 'CAD']['RealGain'].sum().round(2)
    TotRealGainUSD = portfolio[portfolio['Currency'] == 'USD']['RealGain'].sum().round(2)

    # Remove tickers with 0 units
    portfolio = portfolio[portfolio['CumulUnits'] != 0]

    print("Getting online quotes...")
    tickers = portfolio['Ticker'].to_list()
    prices = getPrice(tickers, date)
    portfolio = pd.merge(portfolio, prices, left_on='Ticker', right_on='ticker', how='inner')

    # Portfolio Value
    portfolio['Value'] = portfolio['Adj Close'] * portfolio['CumulUnits']
    TotValueCAD = portfolio[portfolio['Currency'] == 'CAD']['Value'].sum().round(2)
    TotValueUSD = portfolio[portfolio['Currency'] == 'USD']['Value'].sum().round(2)

    # Unrealized Gain/Loss
    portfolio['UnrealGainPerc'] = (portfolio['Value'] - portfolio['SumPurchaseCost']) / portfolio['SumPurchaseCost'] * 100
    portfolio['TotalUnrealGain'] = portfolio['Value'] - portfolio['SumPurchaseCost']
    portfolio.sort_values(by=['TotalUnrealGain'], inplace=True)
    TotUnrealGainCAD = portfolio[portfolio['Currency'] == 'CAD']['TotalUnrealGain'].sum().round(2)
    TotUnrealGainUSD = portfolio[portfolio['Currency'] == 'USD']['TotalUnrealGain'].sum().round(2)

    # Contributions to date
    Contributions = readContributions(ContributionTable, date)
    TotContributionsCAD = Contributions[Contributions['currency'] == 'CAD']['contribution'].sum()
    TotContributionsUSD = Contributions[Contributions['currency'] == 'USD']['contribution'].sum()

    print('\n*** Summary (CAD): ***')
    print('Total Contributions: ${:,.0f}\nTotal Value: ${:,.0f}\nTotal Unrealized Gain: ${:,.0f}\nTotal Realized Gain: ${:,.0f}\n\n'.format(TotContributionsCAD, TotValueCAD, TotUnrealGainCAD, TotRealGainCAD))
    print(portfolio[portfolio['Currency'] == 'CAD'].round(2))

    print('\n*** Summary (USD): ***')
    print('Total Contributions: ${:,.0f}\nTotal Value: ${:,.0f}\nTotal Unrealized Gain: ${:,.0f}\nTotal Realized Gain: ${:,.0f}\n\n'.format(TotContributionsUSD, TotValueUSD, TotUnrealGainUSD, TotRealGainUSD))
    print(portfolio[portfolio['Currency'] == 'USD'].round(2))

    if args.rebalance:
        ExtraInvest = args.rebalance
        broker = args.broker
        currency = args.currency
        rebalance(portfolio, float(ExtraInvest), broker, currency, date)

    if args.dboutput:
        print("Writing to the DB...")
        writeDBValue(date, TotValueCAD, TotContributionsCAD, configurations,
                ResultTableCAD)

        TimeHistoryCAD = timeHistoryRead(configurations, ResultTableCAD, dt.date.today())
        dailydeltaCAD = getDailyVariation(TimeHistoryCAD)

        writeDBValue(date, TotValueUSD, TotContributionsUSD, configurations,
                ResultTableUSD)

        TimeHistoryUSD = timeHistoryRead(configurations, ResultTableUSD, dt.date.today())
        dailydeltaUSD = getDailyVariation(TimeHistoryUSD)

        bodyCAD = 'Daily variation (CAD): $' + str(dailydeltaCAD.round(2)) + '\nTotal value of the portfolio: $' + str(TotValueCAD)
        bodyUSD = 'Daily variation (USD): $' + str(dailydeltaUSD.round(2)) + '\nTotal value of the portfolio: $' + str(TotValueUSD)

        if args.sendmail:
            # sendmail(sender, to, subject, body)
            print("Sending mail to user...")
            sendtext(bodyCAD)
            sendtext(bodyUSD)

    FinishTime = time.time()
    if Timing:
        print("Total Execution Time: ", FinishTime-StartTime)
