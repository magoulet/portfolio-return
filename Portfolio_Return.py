#!/usr/bin/env python3
#
# ToDo:
# - decouple visualization
# - Incorporate Dividends calculation
#
# Notes:
# Filtering a DF: df[df['Type'] == 'Contr']

import argparse
import chart_studio
import chart_studio.plotly as py
import datetime as dt
# ~ from mailer import sendmail
import json
import mysql.connector
import numpy as np
import os
import pandas as pd
import pandas_datareader.data as web
import pandas_market_calendars as mcal
import plotly.express as px
import plotly.graph_objects as go
from pandas.plotting import register_matplotlib_converters
import sys
import time

scriptpath = "/home/pi/projects/tg_bot/"
sys.path.append(os.path.abspath(scriptpath))

from botMsg import telegram_bot_sendtext as sendtext

def getConfigurations():
    path = os.path.dirname(os.path.realpath(sys.argv[0]))
    configurationFile = path + '/config.json'
    configurations = json.loads(open(configurationFile).read())

    return configurations

def ArgParser():
    parser = argparse.ArgumentParser(description='Program Flags')
    parser.add_argument('-o', help="Output to Database (determined in config.json). Otherwise output to stdout only", action="store_true", dest='dboutput', default=False)
    parser.add_argument('-d', help='Snapshot date - format YYYY-MM-DD', action="store", dest='date', default=False)
    parser.add_argument('-m', help='Sends notification message to user', action="store_true", dest='sendmail', default=False)
    parser.add_argument('-rebalance', help='Rebalances Portfolio', action="store", dest='rebalance', default=False)
    args = parser.parse_args()
    
    return args

def read_transactions(SQLtable, date):
    df = DBRead(configurations,SQLtable,date)

    df.sort_values(by=['Date'], ascending=True, inplace=True)

    # Add average cost basis
    df['PurchasePrice'] = df[['Units','Price']].apply(lambda x: x['Units']*x['Price'] if x['Units'] > 0 else 0, axis=1)
    df['Proceeds'] =   df[['Units','Price']].apply(lambda x: -1*x['Units']*x['Price'] if x['Units'] < 0 else 0, axis=1)
    df['CumulUnits'] = df.groupby(['Ticker'])['Units'].cumsum()

    # Walk through df in historical order
    df = df.groupby('Ticker')
    result = pd.DataFrame()

    for name,group in df:
        df2 = df.get_group(name).reset_index(drop=True)

        df2.loc[0, 'SumPurchaseCost'] = df2.loc[0, 'PurchasePrice']
        df2.loc[0, 'AvgCost'] = df2.loc[0, 'SumPurchaseCost'] / df2.loc[0, 'CumulUnits']

        for index, row in df2.iterrows():
            if index > 0:
                if row['Units'] > 0: #Buy order
                    df2.loc[index, 'SumPurchaseCost'] = df2.loc[index, 'PurchasePrice'] + df2.loc[index-1, 'SumPurchaseCost']
                    df2.loc[index, 'AvgCost'] = df2.loc[index, 'SumPurchaseCost'] / df2.loc[index, 'CumulUnits']
                    df2.loc[index, 'CostBasis'] = np.nan
                    df2.loc[index, 'RealGain'] = np.nan

                else: #Sell order
                    df2.loc[index, 'AvgCost'] = df2.loc[index-1,'AvgCost']
                    df2.loc[index, 'SumPurchaseCost'] = df2.loc[index, 'CumulUnits'] * df2.loc[index, 'AvgCost']
                    df2.loc[index, 'CostBasis'] = -1* df2.loc[index, 'AvgCost'] * df2.loc[index, 'Units']
                    df2.loc[index, 'RealGain'] = df2.loc[index, 'Proceeds'] - df2.loc[index, 'CostBasis']

        result = result.append(df2, ignore_index=True, sort=True)
    return result

def marketopen(day): #Check if markets are open at noon on the chosen day
    tsx = mcal.get_calendar('TSX')
    schedule = tsx.schedule(start_date = '2010-01-01', end_date = dt.date.today())
    open = tsx.open_at_time(schedule, pd.Timestamp(day + pd.Timedelta('+9h'), tz='US/Pacific'))
    return open

def dl_pricing_data(df,start,end):
    tickers = df['Ticker'].unique()
    for ticker in tickers:
        try:
            print(ticker)
            quote = web.DataReader(ticker, 'yahoo', start, end)
            quote.to_csv(ticker+'_data.csv')
        except:
            print('{} is missing!'.format(ticker))
            pass

def get_price(tickers,end):
    '''  
    Expected format:  
                        Price
    Attributes Symbols           
    Adj Close  BNS.TO   67.870003
               CAE.TO   33.270000
               ...      ...
    '''
    if OfflineData == True:
        df = pd.read_csv(ticker+'_data.csv', parse_dates = ['Date'], index_col=['Date'])
        lastprice = df.loc[end,:]['Close']
        return lastprice
    else:
        try:
            # ~ breakpoint()
            print("Getting prices for : ", tickers)
            print("Date: ", end)
            quotes = web.DataReader(tickers, 'yahoo', end ,end)
            # quotes = web.DataReader(ticker, 'av-daily', start, end,api_key="0M2357MRIZDYTSJD")
            quotes = quotes.transpose()
            adjclose = quotes.head(len(tickers))
            adjclose.columns = ["Price"]

            return adjclose
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
            data = pd.read_sql(sqlCommand,my_connect)
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
    DatabaseHelper(sqlCommand,"Insert", configurations)
    return None

def DBRead(configurations,table,date):
    sqlCommand = "SELECT * FROM %s WHERE Date <= '%s';" % (table, date)
    data = DatabaseHelper(sqlCommand,"Select", configurations)
    return data

def get_daily_variation(df):
    df['daily variation'] = df['value'].diff()
    dailydelta = df['daily variation'].iloc[-1]
    return df, dailydelta

def plot_results(df):

    # your api key - go to profile > settings > regenerate key
    chart_studio.tools.set_credentials_file(username=username, api_key=api_key)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['date'],y=df['value'],
                        mode='lines',
                        name='Portfolio Value'))

    fig.add_trace(go.Scatter(x=df['date'],y=df['contributions'],
                        mode='lines',
                        name='Total Contributions'))

    fig.update_layout(title='Total Portfolio Value',
                   xaxis_title='Date',
                   yaxis_title='Value (CAD)')

    fig.update_layout(xaxis_range=['2019-09-01',dt.date.today()])
    # fig.update_xaxes(
    #     range=['2019-09-01',dt.date.today()],
    #     constrain="domain"
    # )
    fig.update_yaxes(
        range=[df['value'].truncate(before='2019-09-01').min(),df['value'].max()+10000],
        constrain="domain"
    )

    # fig.show()
    py.plot(fig, filename = 'portfolio', auto_open=False)

def plot_assets_distribution(df):
    df = df.query('Broker == "dob"')
    df.reset_index(inplace=True)
    fig = px.pie(df, values='Value', names='AssetClass', title='Self-Managed Asset Distribution')
    # fig.show()
    py.plot(fig, filename = 'portfolio_distribution', auto_open=False)
    # Reference: https://plotly.com/python/pie-charts/

def rebalance(df,ExtraCash,end):
    print('\nRebalancing Portfolio with ${:,.0f} extra cash\n'.format(ExtraCash))
    # Current weight
    # Sort by AssetClass
    Weight = df.pivot_table(index='AssetClass', values=['Value'], aggfunc=np.sum)
    TargetWeight = pd.read_csv('target_weight.csv')
    Weight = pd.merge(Weight, TargetWeight, on='AssetClass', how='inner')
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
    Weight = pd.merge(Weight,df[['Ticker','Price']],left_on='Ticker',right_on='Ticker',how='inner')

    Weight['NumUnits'] = (Weight['Delta'] / Weight['Price']).round(0)

    print('\nPortfolio Rebalance:')
    print(Weight)

if __name__ == "__main__":
    # ~ register_matplotlib_converters()

    StartTime = time.time()

    configurations = getConfigurations()
    username = configurations["plotly"][0]["user"]
    api_key = configurations["plotly"][0]["api"]
    DownloadOnly = eval(configurations["misc"][0]["DownloadOnly"])
    OfflineData = eval(configurations["misc"][0]["OfflineData"])
    Timing = eval(configurations["misc"][0]["Timing"])
    TransactionTable = configurations["mysql"][0]["TransactionTable"]
    ResultTable = configurations["mysql"][0]["ResultTable"]

    args = ArgParser()

    if args.date:
        try:
            end = dt.datetime.strptime(args.date, "%Y-%m-%d")
        except ValueError:
            msg = "Not a valid date: '{0}'.".format(args.date)
            raise argparse.ArgumentTypeError(msg)
    else:
        end = dt.datetime.combine(dt.date.today(), dt.datetime.min.time())

    if marketopen(end) == False:
        print("Markets closed on the chosen day!")
        quit()

    df = read_transactions(TransactionTable, end)

    portfolio = df.pivot_table(index=['Ticker', 'Broker', 'AssetClass'], values=['CumulUnits','SumPurchaseCost','RealGain'], aggfunc={'CumulUnits':'last','SumPurchaseCost':'last','RealGain':'sum'}, fill_value=0)

    TotRealGain = portfolio['RealGain'].sum().round(2)

    # Remove empty tickers
    portfolio = portfolio[portfolio['CumulUnits'] != 0]

    if DownloadOnly == True:
        dl_pricing_data(df,'2015-01-01',end) # Download all data and write into CSV
        quit()

    print("Getting online quotes...")
    portfolio.reset_index(inplace=True)
    tickers = portfolio['Ticker'].to_list()
    prices = get_price(tickers, end)
    portfolio = pd.merge(portfolio, prices, left_on='Ticker', right_on='Symbols', how='inner')

    # Portfolio Value
    portfolio['Value'] = portfolio['Price'] * portfolio['CumulUnits']
    TotValue = portfolio['Value'].sum().round(2)

    # Unrealized Gain/Loss
    portfolio['UnrealGainPerc'] = (portfolio['Value'] - portfolio['SumPurchaseCost']) / portfolio['SumPurchaseCost'] * 100
    portfolio['TotalUnrealGain']  = portfolio['Value'] - portfolio['SumPurchaseCost']
    portfolio.sort_values(by=['TotalUnrealGain'], inplace=True)
    TotUnrealGain = portfolio['TotalUnrealGain'].sum().round(2)

    # Contributions to date
    Contributions = pd.read_csv('contributions.csv', parse_dates = ['Date'])
    # Remove entries > date
    Contributions = Contributions[Contributions['Date'] <= end]
    TotContributions = Contributions['Contribution'].sum()

    print('Summary: ')
    print('Total Contributions: ${:,.0f}\nTotal Value: ${:,.0f}\nTotal Unrealized Gain: ${:,.0f}\nTotal Realized Gain: ${:,.0f}\n\n'.format(TotContributions,TotValue,TotUnrealGain,TotRealGain))
    print(portfolio.round(2))

    if args.rebalance:
        ExtraInvest = args.rebalance
        rebalance(portfolio,float(ExtraInvest),end)

    if args.dboutput:
        print("Writing to the DB...")
        DBWrite(end, TotValue, TotContributions, configurations, ResultTable)

        TimeHistory = DBRead(configurations,ResultTable,dt.date.today())
        TimeHistory, dailydelta = get_daily_variation(TimeHistory)
        
        print("Plotting Results...")
        plot_results(TimeHistory)
        plot_assets_distribution(portfolio)

        # sender = 'magoulet@gmail.com'
        # to     = 'magoulet@gmail.com'
        # subject = 'Portfolio Value: ' + str(dt.date.today())
        body = 'Daily variation: $' + str(dailydelta.round(2)) + '\nTotal value of the portfolio: $' + str(TotValue)

        if args.sendmail:
            # sendmail(sender, to, subject, body)
            print("Sending mail to user...")
            sendtext(body)

    FinishTime = time.time()
    if Timing:
        print("Total Execution Time: ",FinishTime-StartTime)
