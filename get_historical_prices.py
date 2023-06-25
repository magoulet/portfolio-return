#!/usr/bin/env python3

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
from classes import Security
from main import arg_parser, read_config, build_portfolio, get_price


def dumpHistoricalPrice(portfolio, path):
    for k, v in portfolio.items():
        if os.path.exists(path+v.ticker+'.pickle'):
            with open(path+v.ticker+'.pickle', 'rb') as file:
                df = pickle.load(file)
                try:
                    lastDate = df.index[-1]
                    if (lastDate < pd.Timestamp(v.lastTradeDate)) or (v.lastTradeDate is None):
                        df = pd.concat([df, yf.download(k, start=lastDate, end=v.lastTradeDate)])
                except Exception:
                    print('Error getting date for: {}, lastDate: {}'.format(v.ticker, lastDate))
        else:
            df = yf.download(k, start=v.firstTradeDate, end=v.lastTradeDate)
        with open(path+v.ticker+'.pickle', 'wb') as file:
            pickle.dump(df, file)


def checkPickle(ticker, path):
    try:
        with open(path+ticker+'.pickle', 'rb') as file:
            df = pickle.load(file)
            print(df.head())
    except Exception:
        print('Could not find pickle')

    return None


def createDir(path):
    # Check whether the specified path exists or not
    isExist = os.path.exists(path)

    if not isExist:
        # Create a new directory because it does not exist
        os.makedirs(path)
    return None


if __name__ == "__main__":

    config = read_config()
    dataTable = {'trades': config["mysql"][0]["TransactionTable"]}
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

    createDir(path)

    for currency in currencies:
        portfolio[currency] = build_portfolio(dataTable['trades'], date,
                                              currency, config)
        dumpHistoricalPrice(portfolio[currency], path)
