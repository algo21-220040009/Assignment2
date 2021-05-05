import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xlrd
from WindPy import w
from pycopula.copula import ArchimedeanCopula
from scipy import stats


# 计算下尾相关系数的函数
def CopulaParam_c(df, n):
    a = [0, 1]
    clayton = np.zeros([n, 3])
    archimedean1 = ArchimedeanCopula(family="clayton", dim=2)
    for i in range(1, n):
        clayton0 = archimedean1.fit(df[i - 1:i + 29].values[:, a], method="cmle")
        clayton[i - 1, :][0] = clayton0[0]

    b = [0, 2]
    for i in range(1, n):
        clayton1 = archimedean1.fit(df[i - 1:i + 29].values[:, b], method="cmle")
        clayton[i - 1, :][1] = clayton1[0]
    c = [1, 2]
    for i in range(1, n):
        clayton2 = archimedean1.fit(df[i - 1:i + 29].values[:, c], method="cmle")
        clayton[i - 1, :][2] = clayton2[0]
    return clayton


# 计算上尾相关系数的函数
def CopulaParam_g(df, n):
    a = [0, 1]
    gumbel = np.zeros([n, 3])
    archimedean1 = ArchimedeanCopula(family="gumbel", dim=2)
    for i in range(1, n):
        gumbel0 = archimedean1.fit(df[i - 1:i + 29].values[:, a], method="cmle")
        gumbel[i - 1, :][0] = gumbel0[0]

    b = [0, 2]
    for i in range(1, n):
        gumbel1 = archimedean1.fit(df[i - 1:i + 29].values[:, b], method="cmle")
        gumbel[i - 1, :][1] = gumbel1[0]
    c = [1, 2]
    for i in range(1, n):
        gumbel2 = archimedean1.fit(df[i - 1:i + 29].values[:, c], method="cmle")
        gumbel[i - 1, :][2] = gumbel2[0]
    return gumbel


# 数据回测研究
w.start()
[start, end] = ["20200101", "20210101"]
df = w.wsd(["000001.SH", "931279.CSI", "882415.WI"], "close", start, end, "")
dates = pd.to_datetime(df.Times)

day = dates.strftime("%Y-%m-%d")

df = pd.DataFrame(df.Data).T
df_r = np.log(df / df.shift(1)).dropna()
data0 = df_r.values
n = 213
gumbel = CopulaParam_g(df_r, n)
clayton = CopulaParam_c(df_r, n)
df_gumbel = pd.DataFrame(gumbel)
df_clayton = pd.DataFrame(clayton)

plt.plot(gumbel[:, 0])
plt.plot(clayton[:, 0])
plt.show()

writer = pd.ExcelWriter('copula_result.xlsx')
for i in ['df_r', 'df_gumbel', 'df_clayton']:
    eval(i).to_excel(excel_writer=writer, sheet_name=i, index=True)
writer.save()
writer.close()

