import numpy as np
from sklearn.covariance import EllipticEnvelope
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.ticker as ticker
from datetime import datetime
import matplotlib.dates as mdate
# true_cov = np.array([[1.2,1],[1.545,1],[3,4]])
# X = np.random.RandomState(0).multivariate_normal(mean=[0*len(true_cov)],
#                                                  cov=true_cov,
#                                                  size=500)
tr_data = pd.read_csv('testdata_18.csv')

# print(tr_data)

tr_data=tr_data.fillna(2)
cc=tr_data[['is','power']]
cov = EllipticEnvelope(contamination = 0.019).fit(cc)
pred = cov.predict(cc)
powern=[]
datn=[]
powerab=[]
datab=[]
# inliers are labeled 1 , outliers are labeled -1
normal = cc[pred == 1]
abnormal = cc[pred == -1]


# predict returns 1 for an inlier and -1 for an outlier
# array([ 1, -1])
# cov.covariance_array([[0.7411..., 0.2535...],
#        [0.2535..., 0.3053...]])
# cov.location_array([0.0813... , 0.0427...])
for e in range(len(abnormal)):
    k1=abnormal.iloc[e].name
    datab.append(datetime.strptime(tr_data.loc[k1, 'time'], "%Y/%m/%d"))
    powerab.append(tr_data.loc[k1,'power'])
for e in range(len(normal)):
    k = normal.iloc[e].name
    datn.append(datetime.strptime(tr_data.loc[k, 'time'], "%Y/%m/%d"))
    powern.append(tr_data.loc[k,'power'])

plt.grid(axis="y")
plt.ylim(0,70)
# fig = plt.figure()
font1={'size':18}
# ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
L1,=plt.plot(datab, powerab, 'ro')
L2,=plt.plot(datn, powern, 'bx')
legend=plt.legend([L1,L2],['anomalies','normal data'],loc='upper right',prop = {'size':20})
# legend.get_title().set_fontsize(fontsize = 30)
plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(75))
plt.tick_params(labelsize=18)
# plt.gca().yaxis.set_major_locator(ticker.MultipleLocator(50))
ax=plt.subplot(1,1,1)
ax.xaxis.set_major_formatter(mdate.DateFormatter('%Y-%m'))
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.xlabel('Timeline',font1)
plt.ylabel('Power data',font1)
# plt.show()
all=tr_data[['is']].values.tolist()
alll=[e[0] for e in all ]

# print(alll)
an=0
fp=0
tp=0
fn=0
accur=0
for e in range(len(alll)):
    if pred[e]==-1:
        an+=1
    if alll[e]==pred[e] and pred[e]==-1:
        tp+=1
    if alll[e]!=pred[e] and pred[e]==1:
        fn+=1
    if alll[e] != pred[e] and pred[e] == -1:
        fp+=1
    if alll[e] != pred[e]:
        accur+=1

print("anomalies ",an)
print("precision ",tp/(tp+fp))
print("callback ",tp/(tp+fn))
print("accuracy",1-accur/686)
