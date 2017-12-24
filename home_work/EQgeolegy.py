import pandas as pd
import matplotlib.pyplot as plt

df1 = pd.read_csv('query.csv')
df2 = pd.read_csv('query1990.csv')
df3 = pd.read_csv('Hokkaido.csv')

# df = pd.concat([df1, df2])
df = df3
df.mag = df.mag.astype(int)

df = df.groupby('mag').count()
print df
fig = plt.figure()
ax1 = fig.subplots()
ax1.plot([2,3,4,5,6,7,8], df.time, '-ko')
ax1.set_yscale('log')
plt.title('HOKKAIDO earthquakes')
plt.xlabel('magnitude')
plt.ylabel('count')
plt.show()