import sys
from UI_maindialog import Ui_Dialog
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import pyqtSlot
from FuturesHedge import FutureHedge
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']


class Hedge_GUI(QtWidgets.QMainWindow, Ui_Dialog):
    """
    Class documentation goes here. Class of computing hedge ratio for a specialized date.
    Get input from Dialog GUI, Call <Class FutureHedge> from FuturesHedge.py for inner computing, Set output on Dialog GUI for display
    """

    def __init__(self, parent=None):
        """
        Constructor

        @param parent reference to the parent widget
        @type QWidget
        """
        super(Hedge_GUI, self).__init__(parent)
        self.setupUi(self)

        self.AdjustLot = False


    @pyqtSlot()
    def on_btnno_clicked(self):
        self.Currentlot.setVisible(False)
        self.AdjustLot = False

    @pyqtSlot()
    def on_btnyes_clicked(self):
        self.Currentlot.setVisible(True)
        self.AdjustLot = True

    @pyqtSlot()
    def on_btnclear_clicked(self):
        self.input_index.clear()
        self.input_ttlamount.clear()
        self.input_mgnratio.clear()
        self.input_startdate.clear()
        self.input_icost.clear()
        self.input_fcost.clear()
        self.cur_future1.clear()
        self.cur_future1_contract.clear()
        self.indexlot.clear()
        self.cur_future2.clear()
        self.cur_future2_contract.clear()
        self.cur_future3.clear()
        self.cur_future3_contract.clear()
        self.cur_future4.clear()
        self.cur_future4_contract.clear()
        self.cur_future5.clear()
        self.cur_future5_contract.clear()



    @pyqtSlot()
    def on_btnsubmit_clicked(self):
        self.indexcode = self.input_index.text()
        print("输入品种： ", self.indexcode)
        self.ttlamount = float(self.input_ttlamount.text()[1:])
        print("输入总仓位： ", self.ttlamount)
        self.margin = float(self.input_mgnratio.text())
        print("输入保证金率： ", self.margin)
        self.tradedate = self.input_startdate.text()
        print("输入计算日期： ", self.tradedate)
        self.costi = float(self.input_icost.text()[:-1])/100
        self.costf = float(self.input_fcost.text()[:-1])/100

        if self.indexcode in ['000300.SH','000016.SH','000905.SH']:
            self.regtype = 'Single'   #每次选择单种最优合约
            self.regmethod = 'CGARCH'
        else:
            self.regtype = 'Multiple'  #每次选择单种最优合约
            self.regmethod = 'VAR'

        if self.AdjustLot == True:
            self.OldLots = dict()
            self.OldLots['indexlot'] = float(self.indexlot.text())
            for i in range(5):
                tmp = 'self.cur_future%s.text()' % str(i+1)
                if len(eval(tmp)) > 0:
                    i = str(i+1)
                    exec("self.OldLots['ctt{i}'] = self.cur_future%s.text()" % i)
                    exec("self.OldLots['ctt_q{i}'] = float(self.cur_future%s_contract.text())" % i)


        self.process()
        print('已完成，返回结果至界面')

        # self.label_fig.setPixmap(QtGui.QPixmap(r"C:\Users\Admin\Desktop\长江金工\股指期货对冲\工具\result\Fig\000903.SH_h自由_h最大品种主力_保0.2_ECM.png"))
        # self.label_fig.setScaledContents(True)



    def process(self):
        Main = FutureHedge(self.indexcode, self.ttlamount, self.margin, self.tradedate, self.costi, self.costf)
        if not self.AdjustLot:
            long_amt, initmargin, df_contract, cost1, cost2 = Main.Build(self.regtype, self.regmethod)
            self.out_longamt.setText(str(long_amt))
            self.out_margin.setText(str(initmargin))
            Nctt = df_contract.shape[0]
            text = '建仓操作：\n'
            for i in range(Nctt):

                exec("self.out_future%s.setText(str(df_contract.loc[%s,'contract']))" % (str(i+1),str(i)))
                exec("self.out_future%s_contract.setText(str(df_contract.loc[%s,'shortquantity']))" % (str(i+1), str(i)))
                text = text + '买入合约%s %s张 \n' % (df_contract.loc[i,'contract'], df_contract.loc[i,'shortquantity'])
            text = text + '买入现货￥%s. \n 现货交易成本约为￥%.2f，期货交易成本约为￥%.2f' % (long_amt,cost1,cost2)
            self.TextAdjust.setText(text)


        else:
            long_amt, initmargin, df_select, df_merge, cost1, cost2, long_q, long_dir = Main.Adjust(self.OldLots,self.regtype, self.regmethod)
            self.out_longamt.setText(str(long_amt))
            self.out_margin.setText(str(initmargin))
            Nctt = df_select.shape[0]
            for i in range(Nctt):
                exec("self.out_future%s.setText(str(df_select.loc[%s,'contract']))" % (str(i + 1), str(i)))
                exec("self.out_future%s_contract.setText(str(df_select.loc[%s,'shortquantity']))" % (str(i + 1), str(i)))

            text = '调仓操作：\n'
            for i in range(df_merge.shape[0]):
                text = text + '%s  %s  %s  %s张 \n' % (i+1, df_merge.loc[i,'contract'], df_merge.loc[i,'adj_dir'],df_merge.loc[i,'adj_q'])
            text = text + '%s现货￥%s \n 现货交易成本约为￥%.2f，期货交易成本约为￥%.2f' % (long_dir, long_q,cost1,cost2)
            self.TextAdjust.setText(text)


if __name__ == "__main__":

    app = QtWidgets.QApplication(sys.argv)
    ui = Hedge_GUI()
    ui.show()
    sys.exit(app.exec_())