#!/usr/bin/env python3
#
# Notes:
# Filtering a DF: df[df['Type'] == 'Contr']

import argparse
from botMsg import telegram_bot_sendtext as sendtext
import chart_studio
import chart_studio.plotly as py
import datetime as dt
import json
import mysql.connector
import numpy as np
import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
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
    args = parser.parse_args()

    return args


def read_transactions(SQLtable, date):
    df = DBRead(configurations, SQLtable, date)

    df.sort_values(by=['Date'], ascending=True, inplace=True)

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
        if df.empty:
            print('DataFrame is empty!')
            exit()
        # quotes = web.DataReader(tickers, 'yahoo',  end-dt.timedelta(days=2), end)
        # quotes = web.DataReader(ticker, 'av-daily', start, end,api_key="0M2357MRIZDYTSJD")
        result = df[:1].stack(level=0).rename_axis(['date', 'ticker']).reset_index(level=1)

        return result[['ticker', 'Adj Close']].set_index('ticker')
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


def DBWrite(end, TotValue, TotContributions, configurations, table):
    sqlCommand = "INSERT INTO %s VALUES ('%s', '%s', '%s') ON DUPLICATE KEY UPDATE value=VALUES(value), contributions=VALUES(contributions);" % (table, end, TotValue, TotContributions)
    DatabaseHelper(sqlCommand, "Insert", configurations)
    return None


def DBRead(configurations, table, date):
    sqlCommand = "SELECT * FROM %s WHERE Date <= '%s';" % (table, date)
    data = DatabaseHelper(sqlCommand, "Select", configurations)
    return data


def get_daily_variation(df):
    df['daily variation'] = df['value'].diff()
    dailydelta = df['daily variation'].iloc[-1]
    return df, dailydelta


def plot_results(df):

    # your api key - go to profile > settings > regenerate key
    chart_studio.tools.set_credentials_file(username=username, api_key=api_key)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['date'], y=df['value'],
                             mode='lines',
                             name='Portfolio Value'))

    fig.add_trace(go.Scatter(x=df['date'], y=df['contributions'],
                             mode='lines',
                             name='Total Contributions'))

    fig.update_layout(title='Total Portfolio Value',
                      xaxis_title='Date',
                      yaxis_title='Value (CAD)')

    fig.update_layout(xaxis_range=['2019-09-01', dt.date.today()])
    # fig.update_xaxes(
    #     range=['2019-09-01',dt.date.today()],
    #     constrain="domain"
    # )
    fig.update_yaxes(
        range=[df['value'].truncate(before='2019-09-01').min(), df['value'].max()+10000],
        constrain="domain"
    )

    # fig.show()
    py.plot(fig, filename='portfolio', auto_open=False)


def plot_assets_distribution(df):
    df = df.query('Broker == "dob"')
    df.reset_index(inplace=True)
    fig = px.pie(df, values='Value', names='AssetClass', title='Self-Managed Asset Distribution')
    # fig.show()
    py.plot(fig, filename='portfolio_distribution', auto_open=False)
    # Reference: https://plotly.com/python/pie-charts/


def rebalance(df, ExtraCash, currency, end):
    print('\nRebalancing Portfolio with ${:,.0f} extra cash\n'.format(ExtraCash))
    # Current weight
    # Sort by AssetClass
    df = df[df['Currency'] == currency]
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
    Weight = pd.merge(Weight, df[['Ticker', 'Price']], left_on='Ticker', right_on='Ticker', how='inner')

    Weight['NumUnits'] = (Weight['Delta'] / Weight['Price']).round(4)

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

    df = read_transactions(TransactionTable, date)
    portfolio = df.pivot_table(index=['Ticker', 'Broker', 'AssetClass', 'Currency'], values=['CumulUnits', 'SumPurchaseCost', 'RealGain'], aggfunc={'CumulUnits': 'last', 'SumPurchaseCost': 'last', 'RealGain': 'sum'}, fill_value=0)

    portfolio.reset_index(inplace=True)

    TotRealGainCAD = portfolio[portfolio['Currency'] == 'CAD']['RealGain'].sum().round(2)
    TotRealGainUSD = portfolio[portfolio['Currency'] == 'USD']['RealGain'].sum().round(2)

    # Remove tickers with 0 units
    portfolio = portfolio[portfolio['CumulUnits'] != 0]

    print("Getting online quotes...")
    tickers = portfolio['Ticker'].to_list()
    prices = get_price(tickers, date)
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
    Contributions = pd.read_csv('contributions.csv', parse_dates=['Date'])
    # Remove entries > date
    Contributions = Contributions[Contributions['Date'] <= pd.to_datetime(date)]
    TotContributionsCAD = Contributions[Contributions['Currency'] == 'CAD']['Contribution'].sum()
    TotContributionsUSD = Contributions[Contributions['Currency'] == 'USD']['Contribution'].sum()

    print('\n*** Summary (CAD): ***')
    print('Total Contributions: ${:,.0f}\nTotal Value: ${:,.0f}\nTotal Unrealized Gain: ${:,.0f}\nTotal Realized Gain: ${:,.0f}\n\n'.format(TotContributionsCAD, TotValueCAD, TotUnrealGainCAD, TotRealGainCAD))
    print(portfolio[portfolio['Currency'] == 'CAD'].round(2))

    print('\n*** Summary (USD): ***')
    print('Total Contributions: ${:,.0f}\nTotal Value: ${:,.0f}\nTotal Unrealized Gain: ${:,.0f}\nTotal Realized Gain: ${:,.0f}\n\n'.format(TotContributionsUSD, TotValueUSD, TotUnrealGainUSD, TotRealGainUSD))
    print(portfolio[portfolio['Currency'] == 'USD'].round(2))

    if args.rebalance:
        ExtraInvest = args.rebalance
        currency = args.currency
        rebalance(portfolio, float(ExtraInvest), currency, date)

    if args.dboutput:
        print("Writing to the DB...")
        DBWrite(date, TotValueCAD, TotContributionsCAD, configurations,
                ResultTableCAD)

        TimeHistoryCAD = DBRead(configurations, ResultTableCAD, dt.date.today())
        TimeHistoryCAD, dailydeltaCAD = get_daily_variation(TimeHistoryCAD)

        DBWrite(date, TotValueUSD, TotContributionsUSD, configurations,
                ResultTableUSD)

        TimeHistoryUSD = DBRead(configurations, ResultTableUSD, dt.date.today())
        TimeHistoryUSD, dailydeltaUSD = get_daily_variation(TimeHistoryUSD)

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
