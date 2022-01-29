import numpy as np
from sklearn.covariance import EllipticEnvelope
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.ticker as ticker
# true_cov = np.array([[1.2,1],[1.545,1],[3,4]])
# X = np.random.RandomState(0).multivariate_normal(mean=[0*len(true_cov)],
#                                                  cov=true_cov,
#                                                  size=500)
tr_data = pd.read_csv('testdata-aug.csv')

plt.grid(axis="y")
L1,=plt.plot(tr_data['Time'],tr_data['Power'], 'bo-',linewidth=2)
# plt.legend([L1],['anomalies'],loc='upper right')
plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(5))
# plt.gca().yaxis.set_major_locator(ticker.MultipleLocator(50))
plt.ylim(0,70)
# fig = plt.figure()
font1={'size':18}
# ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
# L1,=plt.plot(datab, powerab, 'ro')
# L2,=plt.plot(datn, powern, 'bx')
# legend=plt.legend([L1,L2],['anomalies','normal data'],loc='upper right',prop = {'size':20})
# # legend.get_title().set_fontsize(fontsize = 30)
# plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(100))
# plt.tick_params(labelsize=18)
# # plt.gca().yaxis.set_major_locator(ticker.MultipleLocator(50))
ax=plt.subplot(1,1,1)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.xlabel('Timeline',font1)
plt.ylabel('Positive Active Power Value',font1)
plt.show()
