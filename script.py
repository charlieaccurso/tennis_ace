import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# load and investigate the data here:
df= pd.read_csv('tennis_stats.csv')
print(df.head())
print(df.info())

# perform exploratory analysis here:
# OFFENSIVE / SERVICE GAME
plt.scatter(df.Aces, df.Winnings)
plt.title('Aces')
plt.show()
plt.clf()
# positive relationship between Aces and Winnings

plt.scatter(df.DoubleFaults, df.Winnings)
plt.title('Double Faults')
plt.show()
plt.clf()
# positive relationship between DoubleFaults and Winnings

plt.scatter(df.BreakPointsFaced, df.Winnings)
plt.title('Break Points Faced')
plt.show()
plt.clf()

# DEFENSIVE / RETURN GAME
plt.scatter(df.BreakPointsOpportunities, df.Winnings)
plt.title('Break Points Opportunities')
plt.show()
plt.clf()
# positive relationship between BreakPointsOpportunities and Winnings

## perform single feature linear regressions here:
# isolate variables (use double brackets)
x= df[['BreakPointsOpportunities']]
y= df[['Winnings']]

# split dataset to train and test
x_train, x_test, y_train, y_test= train_test_split(x, y, train_size=0.8, test_size=0.2)

# create model instance
model= LinearRegression()
# fit model
model.fit(x_train, y_train)
# print model score (R^2)
print(model.score(x_test, y_test))
# get predicted y values
y_predicted= model.predict(x_test)
# scatter plot actual and predicted outcomes
plt.scatter(y_test, y_predicted, alpha=0.4)
plt.title('Model based on Break Points Opportunities')
plt.xlabel('y values TEST')
plt.ylabel('y values PREDICTED')
plt.show()
plt.clf()
#----------------------------------------------------------
x= df[['Aces']]
y= df[['Winnings']]

# split dataset to train and test
x_train, x_test, y_train, y_test= train_test_split(x, y, train_size=0.8, test_size=0.2)

# create model instance
model= LinearRegression()
# fit model
model.fit(x_train, y_train)
# print model score (R^2)
print(model.score(x_test, y_test))
# get predicted y values
y_predicted= model.predict(x_test)
# scatter plot actual and predicted outcomes
plt.scatter(y_test, y_predicted, alpha=0.4)
plt.title('Model based on Aces')
plt.xlabel('y values TEST')
plt.ylabel('y values PREDICTED')
plt.show()
plt.clf()

#-----------------------------------------------------------------
x= df[['BreakPointsFaced']]
y= df[['Winnings']]

# split dataset to train and test
x_train, x_test, y_train, y_test= train_test_split(x, y, train_size=0.8, test_size=0.2)

# create model instance
model= LinearRegression()
# fit model
model.fit(x_train, y_train)
# print model score (R^2)
print(model.score(x_test, y_test))
# get predicted y values
y_predicted= model.predict(x_test)
# scatter plot actual and predicted outcomes
plt.scatter(y_test, y_predicted, alpha=0.4)
plt.title('Model based on Break Points Faced')
plt.xlabel('y values TEST')
plt.ylabel('y values PREDICTED')
plt.show()
plt.clf()

## perform two feature linear regressions here:
x= df[['BreakPointsOpportunities', 'BreakPointsFaced']]
y= df[['Winnings']]

x_train, x_test, y_train, y_test= train_test_split(x, y, train_size=0.8, test_size=0.2)

model= LinearRegression()
model.fit(x_train, y_train)
print(model.score(x_test, y_test))
y_predicted= model.predict(x_test)

plt.scatter(y_test, y_predicted, alpha=0.4)
plt.title('Two-feature model based on Break Points Opportunities and Break Points Faced')
plt.xlabel('y values TEST')
plt.ylabel('y values PREDICTED')
plt.show()
plt.clf()

# -----------------------------------------------
x= df[['BreakPointsOpportunities', 'DoubleFaults']]
y= df[['Winnings']]

x_train, x_test, y_train, y_test= train_test_split(x, y, train_size=0.8, test_size=0.2)

model= LinearRegression()
model.fit(x_train, y_train)
print(model.score(x_test, y_test))
y_predicted= model.predict(x_test)

plt.scatter(y_test, y_predicted, alpha=0.4)
plt.title('Two-feature model based on Break Points Opportunities and Double Faults')
plt.xlabel('y values TEST')
plt.ylabel('y values PREDICTED')
plt.show()
plt.clf()


## perform multiple feature linear regressions here:
x= df[['BreakPointsOpportunities', 'DoubleFaults', 'BreakPointsConverted', 'BreakPointsSaved', 'TotalPointsWon']]
y= df[['Winnings']]

x_train, x_test, y_train, y_test= train_test_split(x, y, train_size=0.8, test_size=0.2)

model= LinearRegression()
model.fit(x_train, y_train)
print(model.score(x_test, y_test))
y_predicted= model.predict(x_test)

plt.scatter(y_test, y_predicted, alpha=0.4)
plt.title('Multiple-feature Model')
plt.xlabel('y values TEST')
plt.ylabel('y values PREDICTED')
plt.show()
plt.clf()
