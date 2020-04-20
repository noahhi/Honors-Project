import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#df = pd.read_pickle('heuristic_results_multiple_no_cycles.pkl')

df = pd.read_excel('heuristic_results_multiple_no_cycles_big.xlsx')

df = df[df['m']==3]
print(df)

opt1 = list(df['option1:time'])
opt2 = list(df['option2:time'])
opt3 = list(df['option3:time'])
opt4 = list(df['option4:time'])
#exact = list(df['exact'])

# get unique values for x,m
unique_xs = df['size'].unique()
# unique_ms = df['dimensions'].unique()
# opts = ['option1','option2','option3','option4','exact']
# ms = []
# for m in unique_ms:
#     by_m = df.loc[df['dimensions']==m]
#     ms.append(list(by_m['time']))
#
N = len(unique_xs)
#
fig, ax = plt.subplots()
width = 5         # the width of the bars

ax.bar(unique_xs-2*width,opt1,width,label="option 1")
ax.bar(unique_xs-width,opt2,width,label="option 2")
ax.bar(unique_xs,opt3,width,label="option 3")
ax.bar(unique_xs+width,opt4,width,label="option 4")
#ax.bar(unique_xs+2*width,exact,width,label="exact")

# #for i in range(len(ms)):
# #    ax.bar(unique_xs+(width*i), ms[i], width, label=f"{unique_ms[i]} dimensions")
#
# ax.bar(unique_xs, )
#
ax.legend()
ax.set_ylabel('Time (s)')
ax.set_xlabel('size (number of variables)')
ax.autoscale_view()
#
plt.show()

#df.to_excel('heuristic_results_multiple_no_cycles.xlsx')
