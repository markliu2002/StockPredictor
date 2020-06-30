# Replace line 16 with the ticker of the stock you want to predict (yahoo finance tickers)
# Replace line 17 with the number of days you want to predict 
# Uncomment lines 113 to 117 depending on your desired model

import pandas as pd # The pandas library can parse CSV files
import pandas_datareader.data as pdr # Functions from pandas_datareader.data extract data from various internet sources into a pandas DataFrame
import numpy as np # numpy allows us to use arrays
from sklearn import preprocessing # sklearn = sci-kit learn (machine learning library). We import preprocessing from it so the macine can easily parse it 
from sklearn.model_selection import train_test_split # train_test_split used to create our training and testing samples. Makes it easy to split/shuffle data
from sklearn.linear_model import LinearRegression # import linear regression algorithm


from sklearn.tree import DecisionTreeRegressor ###################################################################################################################################
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor

from sklearn import svm # import support vector machines, also used for regression
import datetime # import datetime to work with datetime objects
import matplotlib.pyplot as plt # import matplotlib for graphing
from matplotlib import style # import style to style the graphs
plt.style.use('bmh') # using the "bmh" style

ticker = 'AMZN' # Replace this to the ticker of the stock you want to predict (yahoo finance tickers)
prediction_out = 300 # Replace this with the number of days you want to predict




# G E T    D A T A
# collect historical stock data (Jan 1, 2000  -  present)
START = datetime.datetime(2000, 1, 1) # set start data
END = datetime.datetime.now() # end date is present
df = pdr.DataReader( 
    name=ticker, 
    data_source='yahoo', 
    start=START, 
    end=END 
    )

# creates csv file with the data from yahoo
df.to_csv('{}.csv'.format(ticker))

# read in the csv. parse_dates=True so the reader can parse the index as datetime
df = pd.read_csv('{}.csv'.format(ticker), parse_dates=True, index_col=0)

# set df to df but only these columns/features
df = df[['Open', 'High', 'Low', 'Adj Close', 'Volume']] 

# Create a new column/feature for High - Low percent
# High - Low / Low, X 100
df['HL_PCT'] = (df['High']-df['Low'])/df['Low']*100.0 

# Create a new column/feature for daily percent change
# New - Old / Old X 100
df['PCT_change'] = (df['Adj Close']-df['Open'])/df['Open']*100.0

# These are our columns/features
df = df[['Open', 'High', 'Low', 'Adj Close', 'HL_PCT', 'PCT_change', 'Volume']]

# Adj Close is the column/feature that is going to be predicted
prediction_col = 'Adj Close'

# Plot Adj Close in green so user can see difference real Adj Close and predicted Adj Close
df['Adj Close'].plot(color = 'green')

# Replace all NaN elements with 0s, will just be treated as outliers
df.fillna(0)

# Create a new column called label, it is our label(the actual prediction)
# Set it to the 'Adj. Close' column but shifted __ days into the future. Num of days is determined by prediction_out 
df['label'] = df[prediction_col].shift(-prediction_out)




# D E F I N E    F E A T U R E S    &    L A B E L S ( Usually define features as X, and the label that corresponds to the features as y )
# X (features) set to the entire dataframe except the last 'label' column. Converted to numpy array
X = np.array(df.drop(['label'], 1))

# Should do some pre-processing before moving to training and testing. Features should be in a range of -1 to 1 to speed up processing & help with accuracy
# To do this, use the preprocessing module of Scikit-Learn, apply preprocessing.scale to the X variable
X = preprocessing.scale(X) 

# The X_lately variable contains the most recent features, which we're going to predict against 
# SLICE SYNTAX: [start : end : step] => return a new list containing the requested elements
X_lately = X[-prediction_out:]

# Set X equal to X to the point of -prediction_out. [:-prediction_out:] means from index 0 to -prediction_out
X = X[:-prediction_out:] 

# Drop any still NaN's still from the dataframe
# Need this or else will get error of inconsistent numbers
df.dropna(inplace=True)

# y(label) set to the 'label' column in the DataFrame. Also converted to numpy array
y = np.array(df['label'])




# T R A I N I N G    &    T E S T I N G
# The first parameter is the dataset you're selecting to use (X, y)
# train_size sets the size of the training dataset
# test_size sets the size of the testing dataset
X_train , X_test , y_train , y_test = train_test_split(X, y, train_size=0.9, test_size = 0.1)
# The return here is the training set of features, testing set of features, training set of labels, and testing set of labels.




# D E F I N E / T R A I N / T E S T    C L A S S I F I E R
# X_train and y_train are what we used to fit the classifier (classifier = algorithm that implements classification). Set n_jobs to -1, it will use all cores.
clf = LinearRegression(n_jobs=-1)
#clf = SVR(kernel='rbf', C=1e3, gamma=0.1)
#clf = RandomForestRegressor(n_estimators=100)
#clf = GradientBoostingRegressor(n_estimators=200)
#clf = DecisionTreeRegressor() 

# Classifier defined, now train it with sklearn's .fit(). ("fit" = "train")
clf.fit(X_train, y_train)

# Classifier trained, now can test it. **Should train and test on seperate data**. ("score" = "test")
accuracy = clf.score(X_test, y_test)
# print accuracy




# P R E D I C T
# predict with the X_lately variable which contains the most recent features
prediction_set = clf.predict(X_lately) 

# prints the array of predictions (The Adj Close of the next __ days)
print(prediction_set) 

# Prints 0.95-ish accuracy
print(accuracy) 

# Prints num of days to predict
print(prediction_out) 




# V I S U A L I Z E    D A T A
# Ncolumn in DataFrame called 'Prediction', set to NaN . Will put info in it later
df['Prediction'] = np.nan

# Need to grab the last day in the DataFrame and begin assigning each new prediction to a new day 
# this is the very last date and we get the name of it
last_date = df.iloc[-1].name 

# timestamp() function return the time expressed as the number of seconds that have passed since January 1, 1970
last_unix = last_date.timestamp() 

# next_unix is next day. 86400 is seconds in 1 day
next_unix = last_unix + 86400 
# Now we have the next day we wish to use

# Now we add the prediction to the existing DataFrame
for i in prediction_set:
    next_date = datetime.datetime.fromtimestamp(next_unix)
    next_unix = next_unix + 86400 #next_unix set to itself plus the value of 1 day
    df.loc[next_date] = [np.nan for _ in range(len(df.columns)-1)]+[i]
# Iterate through prediction_set, take each prediction and day and then set those as the values in the DataFrame
# Last line sets NaN for all of the columns except the last column, sets the last column to [i] which is the prediction for that day
# df.loc is going to be referencing the index for the DataFrame
# df.loc[next_date] is the datastamp, and the next_date is the index of the DataFrame

# print(df.tail())

# plot 'Prediction' column in red so user can see difference real Adj Close and predicted Adj Close
df['Prediction'].plot(color = 'red')

# Set legend to position 2 (top-left)
plt.legend(loc=2) 

# Title is the ticker
plt.title('{}'.format(ticker))

# X axis label is Date
plt.xlabel('Date')

# Y axis label is Adj Close Price
plt.ylabel('Adj Close Price (USD)')

# Show plot
plt.show()