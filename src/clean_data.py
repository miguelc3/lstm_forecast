import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import os

matplotlib.use('qt5agg')

print(os.getcwd())

path_data = r'./../data/GlobalLandTemperaturesByCountry.csv'

# Load data
data = pd.read_csv(path_data)

# Convert dt column to datetime
data['dt'] = pd.to_datetime(data['dt'])

# Filter relevant columns and rows
data_pt = data[data['Country'] == 'Portugal'].reset_index()
df = data_pt[['dt', 'AverageTemperature']]

# Get data only after 1950
df = df[df['dt'] >= '1950-01-01']
df.set_index('dt', inplace=True)

plt.plot(df.index, df['AverageTemperature'])
plt.title('Average temperature')
plt.xlabel('Date')
plt.ylabel('Temperature')



