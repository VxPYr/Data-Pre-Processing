#1. Importing important libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
#-----------------------------------------------------------------------------------------------------------------------------------------------------
#2. Reading dataset
df = pd.read_csv("Life Expectancy Data.csv")
#head
df.head()
#tail
df.tail()
#-----------------------------------------------------------------------------------------------------------------------------------------------------
#3. Sanity check of data
#shape
df.shape
#info()
df.info()
#finding missing value
# df.isnull().sum()
df.isnull().sum()/df.shape[0]*100
#finding duplicates
df.duplicated().sum()
#-----------------------------------------------------------------------------------------------------------------------------------------------------
#4. Exploratory Data Analysis (EDA)
#identifying garbage values
for i in df.select_dtypes(include= "object").columns:
    print(df[i].value_counts())
    print("***"*10)

#Descriptive Statistics
df.describe().T
df.describe(include='object')

#Histogram to understand distribution
import warnings
warnings.filterwarnings("ignore")
for i in df.select_dtypes(include="number").columns:
    sns.histplot(data=df, x=i)
    plt.show()
    
#Boxplot to identify outliers
for i in df.select_dtypes(include="number").columns:
    sns.boxplot(data=df, x=i)
    plt.show()
    
df.select_dtypes(include="number").columns
df.columns = df.columns.str.strip()
# df.columns = df.columns.str.lower()

#Scatter plot to understand the relationship
for i in ['Year', 'Life expectancy ', 'Adult Mortality', 'infant deaths',
       'Alcohol', 'percentage expenditure', 'Hepatitis B', 'Measles ', ' BMI ',
       'under-five deaths ', 'Polio', 'Total expenditure', 'Diphtheria ',
       ' HIV/AIDS', 'GDP', 'Population', ' thinness  1-19 years',
       ' thinness 5-9 years', 'Income composition of resources', 'Schooling']:
    sns.scatterplot(data=df, x=i, y='Life expectancy')
    plt.show()
    
#correlation with heatmap to interpret the relation and multicollinarity
#df.select_dtypes(include="number").corr()
s = df.select_dtypes(include="number").corr()
plt.figure(figsize=(15,15))
sns.heatmap(s, annot= True)
#-----------------------------------------------------------------------------------------------------------------------------------------------------
#5. Missing values treatment
#choose the method of inputting missing values
#like mean, median, mode or KNNInputer
for i in ["BMI","Polio","Income composition of resources"]:
    df[i].fillna(df[i].median(),inplace=True)
from sklearn.impute import KNNImputer
impute=KNNImputer()
for i in df.select_dtypes(include="number").columns:
    df[i]=impute.fit_transform(df[[i]])
df.isnull().sum()
#-----------------------------------------------------------------------------------------------------------------------------------------------------
#6. Outliers treatment
#Decide whether to do outlier treatment or not.
def wisker(col):
    q1, q3=np.percentile(col, [25,75])
    iqr=q3-q1
    lw=q1-1.5*iqr
    uw=q3+1.5*iqr
    return lw,uw
# wisker()

wisker(df['GDP'])
for i in ['GDP','Total expenditure','thinness  1-19 years','thinness 5-9 years']:
    lw,uw = wisker(df[i])
    df[i] = np.where(df[i]<lw,lw,df[i])
    df[i] = np.where(df[i]>uw,uw,df[i])

for i in ['GDP','Total expenditure','thinness  1-19 years','thinness 5-9 years']:
    sns.boxplot(df[i])
    plt.show()

df.columns
#-----------------------------------------------------------------------------------------------------------------------------------------------------
#7. Duplicates and garbage value treatments
#Check for duplicate if we have any unique column in the data set, delete the garbage values
df.drop_duplicates()
#-----------------------------------------------------------------------------------------------------------------------------------------------------
#8. Encoding of data
#Do label encoding and one hot encoding with pd.getdummies
dummy = pd.get_dummies(data=df, columns= ["Country","Status"], drop_first=True)
dummy