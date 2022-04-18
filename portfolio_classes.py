import math

class Security:
    def __init__(self, ticker):
        self.ticker = ticker
        self.qty = 0
        self.price = 0
        self.moneyIn = 0
        self.realGain = 0
        self.costBasis = 0
        self.logReturn = 0
        self.UnrealizedReturn = 0
        self.totUnrealizedReturn = 0
        self.firstTradeDate = None
        self.lastTradeDate = None



    def buy(self, qty, price, date):

        if qty > 0:
            self.qty += qty
        else:
            print('Cannot buy a negative quantity!')
        self.moneyIn += qty*price
        self.costBasis = self.moneyIn / self.qty

        # A position was closed in the past, then opened again
        if self.firstTradeDate is None:
            self.firstTradeDate = date
        if self.lastTradeDate is not None:
            self.lastTradeDate = None

        # print("Bought {} shares of {} at ${:,} for a total amount of ${:,}".format(qty, self.ticker, price, qty*price))


    def sell(self, qty, price, commission, date):
        # cost basis is constant for a sale transaction
        # Output: realGain, moneyIn

        if qty > 0:
            self.qty -= qty

            # Record date when position was closed
            if self.qty == 0:
                self.lastTradeDate = date
        else:
            print('Cannot sell a negative quantity!')
        self.realGain += qty * (price - self.costBasis)
        self.logReturn += math.log(price/self.costBasis)
        self.moneyIn = self.qty * self.costBasis

        # print("Sold {} shares of {} at ${:,} for a total amount of ${:,}".format(qty, self.ticker, price, qty*price))

    def print(self):
        print('Ticker: {} \nQty: {} \nPrice: {:,} \nMoney In: {:,.0f} \nrealGain: {:,.0f} \nCost Basis: {:,.2f} \nReturn: {:.2f}% \nFirst Trade Date: {} \nLast Trade Date: {}'.format(self.ticker, self.qty, self.price, self.moneyIn, self.realGain, self.costBasis, (math.e**self.logReturn-1)*100, self.firstTradeDate, self.lastTradeDate))

    def split(self, ratio):
        self.qty *= ratio
        self.price *= 1/ratio
