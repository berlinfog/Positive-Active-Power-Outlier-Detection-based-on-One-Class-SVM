from sklearn import svm
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.ticker as ticker
from datetime import datetime
import matplotlib.dates as mdate
# plt.style.use('fivethirtyeight')
from numpy import genfromtxt



# use the same dataset
tr_data = pd.read_csv('testdata_18.csv')

# print(tr_data)

tr_data=tr_data.fillna(1)
clf = svm.OneClassSVM(nu=0.026, kernel='rbf', gamma=0.011)#18 acc 99.4169 call 88.8889
# clf = svm.OneClassSVM(nu=0.026, kernel='rbf', gamma=0.011)#18 88.8889 88.8889 99.4169

#old

'''
OneClassSVM(cache_size=200, coef0=0.0, degree=3, gamma=0.1, kernel='rbf',
      max_iter=-1, nu=0.05, random_state=None, shrinking=True, tol=0.001,
      verbose=False)
'''
cc=tr_data[['is','power']]
# datedate=tr_data[['time']]
# powerpower=tr_data[['power']]
# print(cc)
clf.fit(cc)
pred = clf.predict(cc)
powern=[]
datn=[]
powerab=[]
datab=[]
# inliers are labeled 1 , outliers are labeled -1
normal = cc[pred == 1]
abnormal = cc[pred == -1]
# print(abnormal)
for e in range(len(abnormal)):
    k1=abnormal.iloc[e].name
    datab.append(datetime.strptime(tr_data.loc[k1,'time'],"%Y/%m/%d"))
    # datab.append(tr_data.loc[k1, 'time'])
    # print(tr_data.loc[k,'time'])
    powerab.append(tr_data.loc[k1,'power'])
for e in range(len(normal)):
    k = normal.iloc[e].name
    datn.append(datetime.strptime(tr_data.loc[k, 'time'], "%Y/%m/%d"))
    # datn.append(tr_data.loc[k, 'time'])
    # print(tr_data.loc[k,'time'])
    powern.append(tr_data.loc[k,'power'])

# print(abnormal)
# print(abnormal.loc[34,'deviation'])
# print(abnormal.index[0])
print(len(datab),len(powerab))
# plt.plot(normal.values[:, 0], normal.values[:, 1], 'bx')
# plt.plot(abnormal.values[:, 0], abnormal.values[:, 1], 'ro')
plt.grid(axis="y")
plt.ylim(0,60)
# fig = plt.figure()
font1={'size':18}
# ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
L2,=plt.plot(datn, powern, 'bx')
L1,=plt.plot(datab, powerab, 'ro')

legend=plt.legend([L1,L2],['abnormal data','normal data'],bbox_to_anchor=(1,1),loc='upper right',prop = {'size':15})
# legend=plt.legend([L1,L2],['abnormal data','normal data'],bbox_to_anchor=(0.3,0.98),loc='upper right',prop = {'size':20})
# legend.get_title().set_fontsize(fontsize = 30)
plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(75))
plt.tick_params(labelsize=18)
# plt.gca().yaxis.set_major_locator(ticker.MultipleLocator(50))
ax=plt.subplot(1,1,1)
ax.xaxis.set_major_formatter(mdate.DateFormatter('%Y-%m'))
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.xlabel('Timeline',font1)
plt.ylabel('Positive Active Power Value',font1)
plt.show()
all=tr_data[['is']].values.tolist()
alll=[e[0] for e in all ]

# print(alll)
tru=0
an=0
fp=0
tp=0
fn=0
accur=0
for e in range(len(alll)):
    if alll[e]==-1:
        tru+=1
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
