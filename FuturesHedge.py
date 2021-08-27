import pandas as pd
import numpy as np
from datetime import datetime
import statsmodels.api as sm
from sklearn.linear_model import Lasso, ElasticNet
from sklearn.model_selection import train_test_split
from arch import arch_model
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
from WindPy import w
w.start()


class MarketData(object):
    """
    Class of Obtain data from database as requested.
    """

    database = 'wind'

    def __init__(self, date, indexcode):
        self.indexcode = indexcode
        self.tradedate = date
        self.symbols = ['IF', 'IH', 'IC']
        self.cont_contracts = []
        for symbol in self.symbols:
            self.cont_contracts.extend(
                ['%s00.CFE' % symbol, '%s01.CFE' % symbol, '%s02.CFE' % symbol, '%s03.CFE' % symbol])

    def __call__(self):
        startdate = w.tdaysoffset(-250, self.tradedate, "").Data[0][0]
        futureprice = w.wsd(self.cont_contracts, "settle", startdate, self.tradedate, "Fill=Previous", usedf=True)[1]
        futureprice.index = pd.to_datetime(futureprice.index)
        print("已获取期货价格序列")
        futuretradevol = w.wsd(self.cont_contracts, "volume", startdate, self.tradedate, "Fill=Previous", usedf=True)[1]
        futuretradevol.index = pd.to_datetime(futuretradevol.index)
        print("已获取期货成交量序列")
        indexprice = w.wsd(self.indexcode, "close", startdate, self.tradedate, "Fill=Previous", usedf=True)[1]
        indexprice.index = pd.to_datetime(indexprice.index)
        indexprice.columns = [self.indexcode]
        print("已获取指数价格序列")
        futurecontracts = w.wsd(self.cont_contracts, "trade_hiscode", startdate, self.tradedate, usedf=True)[1]
        futurecontracts.index = pd.to_datetime(futurecontracts.index)
        futurecontracts['date'] = futurecontracts.index.values
        print("已获取期货合约数据")

        all_data = {}
        all_data['fprice'] = futureprice
        all_data['fctt'] = futurecontracts
        all_data['fvol'] = futuretradevol
        all_data['iprice'] = indexprice
        all_data['lsctt'] = self.cont_contracts
        return all_data
        # self.time2expiry = self.futurecontracts.copy()
        # self.lasttradedate = []
        # for i in self.cont_contracts:
        #     contracts = np.unique(self.futurecontracts[i].values)
        #     lastdate = pd.DataFrame(contracts,columns=['code'])
        #     lastdate['lasttrade_date'] = w.wss(list(contracts),'lasttrade_date').Data[0]
        #     lastdate.set_index('code', inplace=True)
        #     self.lasttradedate.append(lastdate)
        #     self.time2expiry[i] = self.time2expiry.apply(
        #         lambda x: w.tdayscount(x['date'],lastdate.loc[x[i],'lasttrade_date']).Data[0][0]-1, axis=1)
        #     print(i)
        # self.lasttradedate = pd.concat(self.lasttradedate).drop_duplicates()


class FutureHedge(object):
    """
    Main function
    """

    def __init__(self, indexcode, TPV, margin, tradedate, icost, fcost):
        self.indexcode = indexcode
        self.tradedate = tradedate
        self.TPV = TPV
        self.margin = margin
        self.margin_am = margin * TPV

        data_object = MarketData(self.tradedate, self.indexcode)
        all_data = data_object()
        print("MarketData运行成功")
        self.indexprice = all_data['iprice']
        self.futureprice = all_data['fprice']
        self.futurecontracts = all_data['fctt']
        self.futuretradevol = all_data['fvol']
        self.symbols = ['IF', 'IH', 'IC']
        self.cont_contracts = all_data['lsctt']

        self.fcost = fcost
        self.icost = icost # 指数交易手续费
        self.multiplier = 300

        self.long_price = 0  # 做多点
        self.long_position = 0
        self.long_endbal = 0  # 多头期末仓位
        self.short_price = 0  # 做空点
        self.short_position = 0  # 空头仓位
        self.short_chg = 0  # 空头仓位变化量
        self.margin_endbal = 0  # 保证金期末仓位
        self.spread = 0  # 跨期价差 = 新合约价格 - 旧合约价格
        self.basis = 0  # 基差 = 现货价格 - 期货价格
        self.contract = ''
        self.HedgeRecords = []

    def Build(self, regtype, regmethod):

        self.long_position = (self.TPV - self.margin_am) * (1 - self.icost)  # 多头仓位
        cost1 = (self.TPV - self.margin_am) * self.icost
        if regtype == 'Single':
            pass

        else:
            select_ctt = self.HedgeRatio(self.indexprice, self.futureprice, regtype, regmethod=regmethod).reset_index()
            select_ctt.columns = ['cont_ctt', 'hedgeratio']
            print('对冲比例 - %s \n %s' % (self.tradedate, select_ctt))
            select_ctt['contract'] = select_ctt['cont_ctt'].map(lambda x: self.futurecontracts.loc[self.tradedate, x])
            select_ctt['shortprice'] = select_ctt['cont_ctt'].map(lambda x: self.futureprice.loc[self.tradedate, x])
            select_ctt['multiplier'] = select_ctt['cont_ctt'].map(lambda x: 200 if x[:2] == 'IC' else 300)
            select_ctt['shortquantity'] = np.round(select_ctt['hedgeratio'] * self.long_position / (
                    select_ctt['multiplier'] * select_ctt['shortprice']))
            select_ctt['shortposition'] = select_ctt['shortquantity'] * select_ctt['multiplier'] * select_ctt[
                'shortprice']
            select_ctt = select_ctt[select_ctt['shortposition'] != 0]
            self.margin_am = self.margin_am - select_ctt['shortposition'].sum() * self.fcost
            cost2 = select_ctt['shortposition'].sum() * self.fcost

            return self.long_position, self.margin_am, select_ctt[['contract','shortquantity']], cost1, cost2


    def Adjust(self, oldlot, regtype, regmethod):

        """
        :param oldlot: dict(); keys: ['indexlot', 'ctt1', 'ctt1_q','ctt2', 'ctt2_q'...]
        :param regtype: 'Single'/'Multiple'
        :param regmethod:
        :return: long position, initial margin, new contract lots
        """

        # 调整多头仓位
        OldLot = pd.Series(oldlot)
        self.long_position = self.TPV - self.margin_am
        Old_long = OldLot['indexlot']
        self.long_position = self.long_position - abs(self.long_position - Old_long)*self.icost
        long_q = self.long_position - Old_long
        long_dir = '买入' if (self.long_position - Old_long)>0 else '卖出'
        cost1 = abs(self.long_position - Old_long)*self.icost #现货交易手续费

        Nctt = int((len(OldLot)-1) / 2)
        Old_ctt = pd.DataFrame(index=range(Nctt),columns=['contract','oldquantity'])
        for i in range(Nctt):
            Old_ctt.iloc[i,0] = OldLot['ctt{i}']
            Old_ctt.iloc[i,1] = OldLot['ctt_q{i}']
        Old_ctt['oldprice'] = Old_ctt['contract'].map(lambda x: w.wsd(x,"settle",self.tradedate,self.tradedate).Data[0][0])
        Old_ctt['multiplier'] = Old_ctt['contract'].map(lambda x: 200 if x[:2] == 'IC' else 300)
        Old_ctt['oldposition'] = Old_ctt['oldquantity'] * Old_ctt['multiplier'] * Old_ctt['oldprice']
        print(Old_ctt)

        if regtype == 'Single':
            pass
        
        else:
            select_ctt = self.HedgeRatio(self.indexprice, self.futureprice, regtype, regmethod=regmethod).reset_index()
            select_ctt.columns = ['cont_ctt', 'hedgeratio']
            print('对冲比例 - %s \n %s' % (self.tradedate, select_ctt))
            select_ctt['contract'] = select_ctt['cont_ctt'].map(lambda x: self.futurecontracts.loc[self.tradedate, x])
            select_ctt['shortprice'] = select_ctt['cont_ctt'].map(lambda x: self.futureprice.loc[self.tradedate, x])
            select_ctt['multiplier'] = select_ctt['cont_ctt'].map(lambda x: 200 if x[:2] == 'IC' else 300)
            select_ctt['shortquantity'] = np.round(select_ctt['hedgeratio'] * self.long_position / (
                    select_ctt['multiplier'] * select_ctt['shortprice']))
            select_ctt['shortposition'] = select_ctt['shortquantity'] * select_ctt['multiplier'] * select_ctt[
                'shortprice']
            select_ctt = select_ctt[select_ctt['shortposition'] != 0]

            Merge_ctt = select_ctt.merge(Old_ctt,on='contract',how='outer').fillna(0)
            Merge_ctt['adj_am'] = Merge_ctt['shortposition'] - Merge_ctt['oldposition']
            Merge_ctt['adj_q'] = Merge_ctt['shortquantity'] - Merge_ctt['oldquantity']
            Merge_ctt['adj_dir'] = Merge_ctt['adj_q'].map(lambda x: '买' if x>0 else '卖') #1开仓，0平仓
            print(Merge_ctt)
            cost2 = np.sum(np.abs(Merge_ctt['adj_am'])) * self.fcost #期货交易手续费

            self.margin_am = self.margin_am - cost2

        return self.long_position, self.margin_am, select_ctt, Merge_ctt, cost1, cost2,long_q, long_dir





    def HedgeRatio(self, index, futures, regtype='Multiple', regmethod='VAR'):
        '''
        Function of calculate optimal hedge ratio.
        :param regtype - Single(return single contract) /
                          Multiple(return multiple contracts and corresponding ratio)
        :param regmethod - Single{'OLS', 'ECM', 'CGARCH'}
                            Multiple{'OLS', 'VAR'}
        '''
        r_index = np.log(index/index.shift(1)).dropna()
        r_future = np.log(futures/futures.shift(1)).dropna()
        # r_index = r_index[(r_index <= 0.09) & (r_index >= -0.09)]  # 剔除极端值
        # r_future = r_future[(r_future <= 0.09) & (r_future >= -0.09)].fillna(method='ffill')
        # r_merge = pd.merge(r_index, r_future, left_index=True, right_index=True)
        # r_index, r_future = r_merge.iloc[:, 0], r_merge.iloc[:, 1]

        if regtype == 'Single':
            if regmethod == 'OLS':  # OLS回归
                X = sm.add_constant(r_future)
                results = sm.OLS(r_index, X).fit()
                beta = results.params[1]
            elif regmethod == 'ECM':  # Error Correction Model
                future_1 = sm.add_constant(futures)
                price_ols = sm.OLS(index, future_1).fit()
                res = index.values - price_ols.predict()
                X = pd.DataFrame(r_future)
                X['res'] = res[:-1]
                X = sm.add_constant(X)
                results = sm.OLS(r_index, X).fit()
                beta = results.params[1]
            elif regmethod == 'CGARCH':
                future_1 = sm.add_constant(futures)
                price_ols = sm.OLS(index, future_1).fit()
                res = index.values - price_ols.predict()
                res_arch_index = arch_model(r_index, res[:-1], mean='Constant', lags=1, p=1, o=0, q=1,
                                            rescale=False).fit()
                res_arch_future = arch_model(r_future, res[:-1], mean='Constant', lags=1, p=1, o=0, q=1,
                                             rescale=False).fit()
                vol_index = res_arch_index.forecast(horizon=21).variance.values[-1]
                vol_future = res_arch_future.forecast(horizon=21).variance.values[-1]
                e_index = res_arch_index.resid.values[-60:]
                e_future = res_arch_future.resid.values[-60:]
                c_corr = np.corrcoef(e_index, e_future)[1][0]
                beta = (c_corr * np.sqrt(vol_index) / np.sqrt(vol_future)).mean()

            if beta < 0.5:
                beta = 0.5
            elif beta > 2:
                beta = 2
            select_ctt = pd.Series(beta, index=r_future.columns)

        else:
            if regmethod == 'OLS':
                # print(r_future.corr())
                X_train, X_test, y_train, y_test = train_test_split(r_future, r_index, test_size=0.1)
                lasso_model = ElasticNet(0.000001, 1.0)
                lasso_model.fit(X_train, y_train)
                coefs = pd.Series(lasso_model.coef_, index=self.cont_contracts)
                # print(coefs)
                select_ctt = coefs[coefs != 0]
                print('1\n', select_ctt)
                while np.any(select_ctt < 0) or len(select_ctt) > 4:
                    reduce_ctt = select_ctt[select_ctt > 0]
                    X_train_2, X_test_2, y_train_2, y_test_2 = train_test_split(r_future[reduce_ctt.index], r_index,
                                                                                test_size=0.1)
                    lasso_model_2 = ElasticNet(0.000001, 1.0)
                    lasso_model_2.fit(X_train_2, y_train_2)
                    coefs_2 = pd.Series(lasso_model_2.coef_, index=reduce_ctt.index)
                    print('2\n', coefs_2)
                    select_ctt = coefs_2[coefs_2 != 0]

            elif regmethod == 'VAR':
                lag_all = pd.concat([r_index, r_future], axis=1).shift(1).dropna()
                X_index = sm.add_constant(lag_all)
                res_index = sm.OLS(r_index.values[1:], X_index.values).fit().resid  # 指数向量自回归残差
                res_futures = np.ndarray(shape=(lag_all.shape[0], r_future.shape[1]))
                for col in range(len(r_future.columns)):
                    X_future = sm.add_constant(lag_all)
                    res_future = sm.OLS(r_future.iloc[1:, col], X_future.values).fit().resid
                    res_futures[:, col] = res_future  # 期货（自变量）向量自回归残差
                X_train, X_test, y_train, y_test = train_test_split(res_futures, res_index, test_size=0.1)
                lasso_model = ElasticNet(0.000001, 1.0)
                lasso_model.fit(X_train, y_train)
                coefs = pd.Series(lasso_model.coef_, index=self.cont_contracts)
                # print(coefs)
                select_ctt = coefs[coefs != 0]
                print('1\n', select_ctt)
                while np.any(select_ctt < 0) or len(select_ctt) > 4:
                    reduce_ctt = select_ctt[select_ctt > 0]
                    X_train_2, X_test_2, y_train_2, y_test_2 = train_test_split(r_future[reduce_ctt.index], r_index,
                                                                                test_size=0.1)
                    lasso_model_2 = ElasticNet(0.000001, 1.0)
                    lasso_model_2.fit(X_train_2, y_train_2)
                    coefs_2 = pd.Series(lasso_model_2.coef_, index=reduce_ctt.index)
                    print('2\n', coefs_2)
                    select_ctt = coefs_2[coefs_2 != 0]

        return select_ctt
