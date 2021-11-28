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
        self.totUnrealizedReturn= 0


    def buy(self, qty, price):

        if qty > 0:
            self.qty += qty
        else:
            print('Cannot buy a negative quantity!')
        self.moneyIn += qty*price
        self.costBasis = self.moneyIn / self.qty

        # print("Bought {} shares of {} at ${:,} for a total amount of ${:,}".format(qty, self.ticker, price, qty*price))


    def sell(self, qty, price, commission):
        # cost basis is constant for a sale transaction
        # Output: realGain, moneyIn

        if qty > 0:
            self.qty -= qty
        else:
            print('Cannot sell a negative quantity!')
        self.realGain += qty * (price - self.costBasis)
        self.logReturn += math.log(price/self.costBasis)
        self.moneyIn = self.qty * self.costBasis

        # print("Sold {} shares of {} at ${:,} for a total amount of ${:,}".format(qty, self.ticker, price, qty*price))

    def print(self):
        print('Ticker: {} \nQty: {} \nPrice: {:,} \nMoney In: {:,.0f} \nrealGain: {:,.0f} \nCost Basis: {:,.2f} \nReturn: {:.2f}%'.format(self.ticker, self.qty, self.price, self.moneyIn, self.realGain, self.costBasis, (math.e**self.logReturn-1)*100))

    # def split(self, ratio):
