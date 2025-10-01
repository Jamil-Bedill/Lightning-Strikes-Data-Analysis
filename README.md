# Lightning Strikes Data Analysis
## Overview

This project is part of the Google Advanced Data Analytics course on Coursera. It focuses on exploring a dataset of lightning strikes, detecting outliers, dealing with missing values, and applying encoding techniques to prepare the data for analysis and visualisation.

## Objectives

- Handle missing data and outliers in real-world datasets.
- Perform descriptive statistics and visualisations.
- Encode categorical variables for analysis.
- Use Python libraries (pandas, numpy, matplotlib, seaborn) for data wrangling and visualisation.

## Dataset

The analysis uses sample datasets (eda_outliers_dataset1.csv, eda_outliers_dataset2.csv, and eda_outliers_dataset3.csv) provided in the course.
  - **Features include**: year, date, number_of_strikes, and center_point_geom.
  - Covers multiple years of recorded lightning strikes.
## Key Steps in the Analysis
### 1. Data Cleaning
- Importing the relevant libraries and loading the dataset
- Did some feature engineering.
- Found mean and median
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
```
Now loading the dataset and showing the first few rows of the dataset. 
```
df = pd.read_csv('/content/eda_outliers_dataset1.csv')
df.head()
```

<img width="235" height="195" alt="image" src="https://github.com/user-attachments/assets/30f6a9f4-1c71-47e1-ba6c-bcb82cfeb4ab" />

The number seems quite difficult to read. We add another column and change it to readable numbers that are easy to read.

```
def readable_num1(x):
  if x>=1e6:
     s = f'{x*1e-6:.1f}M'
  else:
    s = f'{x*1e-3:.0f}K'
  return s
df['number_of_strikes_readable']= df['number_of_strikes'].apply(readable_num1)
df.head()
```

<img width="469" height="189" alt="image" src="https://github.com/user-attachments/assets/8331203e-1350-4c8e-862a-fbf75caf145e" />

We can see that the numbers are now easy to read. Let's find the mean and median of number_of_strikes and display the result in terms of readable numbers. 

```
print(f'Mean: {readable_num1(df['number_of_strikes'].mean())}')
print(f'Median: {readable_num1(df['number_of_strikes'].median())}')
```

Mean: 26.8M
Median: 28.3M
### 2. Outlier Detection
Plotted a box plot to detect outliers
Used the Inter Quantile Range IQR to find the outliers
Let us create a box plot to detect outliers

```
box = sns.boxplot(x=df['number_of_strikes'])
g = plt.gca()
box.set_xticklabels(np.array([readable_num1(x) for x in g.get_xticks()]))
plt.xlabel('Number of strikes')
plt.title('Yearly number of lightning strikes')
plt.show()
```

<img width="584" height="463" alt="image" src="https://github.com/user-attachments/assets/b1b26141-8a27-41e0-b53c-36ff34da3777" />

We can see that the dataset has outliers. Two data points are below 10 million. To identify outliers, first calculate the first and third quartiles (Q1 and Q3) and the interquartile range (IQR). Any data point more than 1.5 times the IQR below Q1 or above Q3 is considered an outlier.

```
percentile_25 = df['number_of_strikes'].quantile(0.25)
percentile_75 = df['number_of_strikes'].quantile(0.75)
iqr = percentile_75 - percentile_25
# Now finding the lower and upper limit
upper_limit = percentile_75 +1.5*iqr
lower_limit = percentile_25 -1.5*iqr
# Displaying upper and lower limits.
print(f'The upper limit is {readable_num1(upper_limit)}')
print(f'The lower limit is {readable_num1(lower_limit)}')
```

The upper limit is 47.4M
The lower limit is 8.6M
Now we filter the dataset and remove any data that is less than the lower limit. We don't filter based on the upper limit because in the boxplot, we did not notice any outlier towards that side. 

```
df[df['number_of_strikes']<lower_limit]
```

<img width="515" height="101" alt="image" src="https://github.com/user-attachments/assets/7e60bb43-800a-4e19-a020-e31088085fc2" /> 

There are two outliers: One is 1987, and the other is 2019. This dataset does not have data for each year. We will now load the 2019 data and check for any anomalies. And then will check the 1987 dataset. 

```
# Let's get the 2019 dataset
df_2019 = pd.read_csv('/content/eda_outliers_dataset2.csv')
df_2019.head()
```

<img width="444" height="197" alt="image" src="https://github.com/user-attachments/assets/0ac151a0-a507-49aa-9fd2-dff7ed2a6e04" />

We now change the date column to datetime and add two columns: one month number and the other month name

```
df_2019['date']= pd.to_datetime(df_2019['date'])
df_2019['month']= df_2019['date'].dt.month
df_2019['month_text'] = df_2019['date'].dt.month_name().str.slice(stop = 3)
df_2019.head()
```

<img width="634" height="189" alt="image" src="https://github.com/user-attachments/assets/36e57323-e3c7-459a-ad49-cfb4e78c0e65" />


Let's explore the dataset. We group the number of strikes by month and month name, and find the sum of strikes for each month. 

```
df_2019_by_month = df_2019.groupby(['month','month_text'])['number_of_strikes']\
    .sum().reset_index().sort_values('month', ascending=True)
df_2019_by_month
```

<img width="349" height="69" alt="image" src="https://github.com/user-attachments/assets/c2d0d72e-d46a-4627-825c-e62d497aefa2" />

The data for 2019 is incomplete because it is only for December. 
Let's check the data for 1987. We load the dataset and follow the same procedure as above. 

```
df_1987 = pd.read_csv('/content/eda_outliers_dataset3.csv')
df_1987.head(10)
```

<img width="447" height="356" alt="image" src="https://github.com/user-attachments/assets/5bfaa125-eab6-4d26-bed2-0806e95f984b" />

Now we change the date column to datetime and add a month and month name column, and finally group the data by month and month name and find the sum. 

```
df_1987['date']= pd.to_datetime(df_1987['date'], format ='%Y-%m-%d')
df_1987['month']= df_1987['date'].dt.month
df_1987['mont_txt']= df_1987['date'].dt.month_name()
df_1987.groupby(['month', 'mont_txt'])['number_of_strikes'].sum().reset_index().sort_values('month', ascending=True)
```

<img width="340" height="402" alt="image" src="https://github.com/user-attachments/assets/8857cf15-04ac-4dc4-9acc-2d0f620be167" />

The dataset is complete. 
Now we can filter out the 2019 data, which was an outlier in the first dataset, and keep the 1987 data. And then check the mean and median. 

```
df_without_outlier = df[df['number_of_strikes']>= lower_limit]
print(f'Mean: {readable_num1(df_without_outlier['number_of_strikes'].mean())}')
print(f'Median: {readable_num1(df_without_outlier['number_of_strikes'].median())}')
```

Mean: 28.2M
Mean: 28.8M

We can see that by detecting and removing the outlier change the mean and median of the dataset. With outliers, the mean was 26.8M and the median was 28.3M. After removing the outlier, the mean is 28.2 M, and the median is 28.8 M. 





