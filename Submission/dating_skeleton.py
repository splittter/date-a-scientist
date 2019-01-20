import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from datetime import datetime
from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import precision_recall_fscore_support as pr
from sklearn.svm import SVC
from sklearn.linear_model import LinearRegression

#Create the df
df = pd.read_csv("profiles.csv")

#Look at columns involved
print(list(df))

#list values in a column in a nice order e.g. age
religions = df.religion.dropna().unique()
religions.sort()
for i in religions:
    print(i)

#plot a chart of ages
ages = []
counts = []
for i in df.age.dropna().unique():
    ages.append(i)
    counts.append(df.age.dropna().value_counts()[i])

plt.bar(ages, counts)
plt.xlabel("Age")
plt.ylabel("Number of Users")
plt.show()

#create three new columns for drinks, drugs and smokes
drink_mapping = {"not at all": 0, "rarely": 1, "socially": 2, "often": 3, "very often": 4, "desperately": 5}
drugs_mapping = {"never": 0, "sometimes": 1, "often": 2}
smokes_mapping = {"no": 0, "when drinking": 1, "trying to quit": 2, "sometimes": 3, "yes": 4}
df["drinks_code"] = df.drinks.map(drink_mapping)
df["drugs_code"] = df.drugs.map(drugs_mapping)
df["smokes_code"] = df.smokes.map(smokes_mapping)

#create a column that shows how many days ago someone logged in - assumes the latest last_online is the day the data was created
def how_long(row, comp):
    row_date_str = row['last_online']
    if not pd.isnull(row_date_str):
        row_date = datetime.strptime(row_date_str[:10], '%Y-%m-%d')
        diff = comp - row_date
        return diff.days
    else:
        return np.nan

date_format = "%Y-%m-%d"
days = []
for i in df.last_online.dropna().unique():
    days.append(datetime.strptime(i[:10], date_format))
max_day = max(days)

df['last_online_code'] = df.apply(how_long, comp=max_day, axis=1)

#create a column for the number of children - 0 for none, 1 for one child and 2 for more than one child
def kids(row):
    kid_str = row['offspring']
    if not pd.isnull(kid_str):
        if kid_str[:5] == 'has a':
            return 1
        elif kid_str[:5] == 'has k':
            return 2
        else:
            return 0
    else:
        return np.nan

df['offspring_code'] = df.apply(kids, axis=1)

#create a column for how good at languages someone is - awards 3 points for each fluent language, 2 for being okay and 1 for being poor
#assumes no qualification of level is fluent as the largest response is english with no qualification for level, and I assume the data is from an english speaking country
def langs(row):
    lang_str = row['speaks']
    if not pd.isnull(lang_str):
        lang_score = 0
        for i in row['speaks'].split(', '):
            lang_level = i.split('(')
            if len(lang_level) > 1:
                if lang_level[1][0] == 'p':
                    lang_score += 1
                elif lang_level[1][0] == 'o':
                    lang_score += 2
                else:
                    lang_score += 3
            else:
                lang_score += 3
        return lang_score
    else:
        return np.nan

df['speaks_code'] = df.apply(langs, axis=1)

#create a df that only includes people who follow some religion
non_religious = ['agnosticism', 'agnosticism and laughing about it', 'agnosticism and somewhat serious about it', 'agnosticism and very serious about it', 'agnosticism but not too serious about it', 'atheism', 'atheism and laughing about it', 'atheism and somewhat serious about it', 'atheism and very serious about it', 'atheism but not too serious about it']
religious_df = df[~df.religion.isin(non_religious)]

#create a new column that groups people by the seriousness of the religion, regardless of the religion, then chart it
def seriousness(row):
    if "laughing" in row['religion']:
        return 0
    if "not too" in row['religion']:
        return 1
    if "somewhat" in row['religion']:
        return 3
    if "very" in row['religion']:
        return 4
    else:
        return 2
    
religious_df['rel_code'] = religious_df.apply(seriousness, axis=1)

x = list(range(5))
y = [religious_df.rel_code.value_counts()[i] for i in x]
plt.bar(x, y)
plt.xlabel("Seriousness")
plt.ylabel("Number of people")
plt.show()

###try to use a KNeighboursClassifier to predict religious seriousness from drinks, drugs and smokes codes
religious_df.dropna(inplace=True, subset=['drinks_code', 'drugs_code', 'smokes_code'])

rel_labels = religious_df['rel_code']
#I tried different sets of data
rel_data = religious_df[['drinks_code', 'drugs_code', 'smokes_code']]
#rel_data = religious_df[['drinks_code', 'drugs_code']]
scaled_rel_data = scale(rel_data)

training_data, validation_data, training_labels, validation_labels = train_test_split(scaled_rel_data, rel_labels, test_size = 0.2, random_state=4949)

#loop to check accuracies etc.
rel_accuracies = []
rel_precisions = []
rel_recalls = []
for i in range(1, 51):
    classifier = KNeighborsClassifier(n_neighbors = i)
    classifier.fit(training_data, training_labels)
    prfs = pr(validation_labels, classifier.predict(validation_data), average='weighted')
    rel_accuracies.append(classifier.score(validation_data, validation_labels))
    rel_precisions.append(prfs[0])

#charts accuracies etc.
k_list = list(range(1, 51))
plt.plot(k_list, rel_accuracies)
plt.plot(k_list, rel_precisions)
plt.xlabel("k")
plt.title("Religious Seriousness Classifier Scores")
plt.show()

#check predictions of best K
classifier = KNeighborsClassifier(n_neighbors = 32)
classifier.fit(training_data, training_labels)
predictions = classifier.predict(validation_data)
prfs = pr(validation_labels, predictions, average='weighted')
print(classifier.score(validation_data, validation_labels))
print(prfs[0])
scores=[0 for _ in range(5)]
for j in predictions:
    scores[j] += 1
plt.bar(list(range(5)), scores)
plt.xlabel("Predicted Seriousness")
plt.ylabel("Number of Predictions")
plt.show()

###see if a SVM does better at predicting religious seriousness from drinks, drugs and smokes codes
rel_labels = religious_df['rel_code']
rel_data = religious_df[['drinks_code', 'drugs_code', 'smokes_code']]
#rel_data = religious_df[['drinks_code', 'drugs_code']]
scaled_rel_data = scale(rel_data)

training_data, validation_data, training_labels, validation_labels = train_test_split(scaled_rel_data, rel_labels, test_size = 0.2, random_state=4949)

#setup to record accuracy and precsion of various gamma and C values
rel_accuracies = []
rel_precisions = []

#check a sepcific gamma and C value with the classifier
#I ended up running this code by hand each time as it was causing my aging machine to overheat in a loop
gam = 1
c = 1
classifier = SVC(kernel='rbf', gamma=gam, C=c)
classifier.fit(training_data, training_labels)
prfs = pr(validation_labels, classifier.predict(validation_data), average='weighted')
rel_accuracies.append([(gam, c), classifier.score(validation_data, validation_labels)])
rel_precisions.append([(gam, c), prfs[0]])

#print the accuracies and precisions for the 'amount' of trials of the SVM you have run
rel_accuracies.sort(key=lambda x:x[1], reverse=True)
rel_precisions.sort(key=lambda x:x[1], reverse=True)
amount = range(10)
for i in amount:
    print(rel_accuracies[i])
print("--")
for i in amount:
    print(rel_precisions[i])

#check predictions for best Gamma and C
classifier = SVC(kernel='rbf', gamma=1, C=1)
classifier.fit(training_data, training_labels)
predictions = classifier.predict(validation_data)
prfs = pr(validation_labels, predictions, average='weighted')
print(classifier.score(validation_data, validation_labels))
print(prfs[0])
scores=[0 for _ in range(5)]
for j in predictions:
    scores[j] += 1
plt.bar(list(range(5)), scores)
plt.xlabel("Predicted Seriousness")
plt.ylabel("Number of Predictions")
plt.show()

###See if a linerarregressor can predict age based on 5 attributes
all_age_df = df.dropna(subset=['age', 'last_online_code', 'offspring_code', 'drinks_code', 'drugs_code', 'speaks_code'])
y = all_age_df[['age']]
X = all_age_df[['last_online_code', 'offspring_code', 'drinks_code', 'drugs_code', 'speaks_code']]
#At one point I looked to see if removing offspring_code helped
#X = all_age_df[['last_online_code', 'drinks_code', 'drugs_code', 'speaks_code']]
x_scale = scale(X)
x_train, x_test, y_train, y_test = train_test_split(x_scale, y, test_size = 0.2, random_state=4949)
lm = LinearRegression()
lm.fit(x_train, y_train)
y_predict = lm.predict(x_test)

#Plot a chart showing predicted ages against actual ages
plt.scatter(y_test, y_predict, alpha=0.4)
plt.xlabel('Actual Age')
plt.ylabel('Predicted Age')
plt.title('Actual Vs Predicted Age')
plt.show()

#print the coeficients of the features and the R2 score for the test data
print(lm.coef_)
print(lm.score(x_test, y_test))


###See if resticting the ages we're interested in to those with the most samples helps
twen_fortytwo = list(range(20,43))
twen_fortytwo_df = all_age_df[all_age_df.age.isin(twen_fortytwo)]
y = twen_fortytwo_df[['age']]
X = twen_fortytwo_df[['last_online_code', 'offspring_code', 'drinks_code', 'drugs_code', 'speaks_code']]
#X = twen_fortytwo_df[['last_online_code', 'drinks_code', 'drugs_code', 'speaks_code']]
x_scale = scale(X)
x_train, x_test, y_train, y_test = train_test_split(x_scale, y, test_size = 0.2, random_state=4949)
lm = LinearRegression()
lm.fit(x_train, y_train)
y_predict = lm.predict(x_test)

plt.scatter(y_test, y_predict, alpha=0.4)
plt.xlabel('Actual Age')
plt.ylabel('Predicted Age')
plt.title('Actual Vs Predicted Age')
plt.show()

print(lm.coef_)
print(lm.score(x_test, y_test))


###See if a KNeighboursRegressor does better at age predicting
y = all_age_df[['age']]
X = all_age_df[['last_online_code', 'offspring_code', 'drinks_code', 'drugs_code', 'speaks_code']]
#X = all_age_df[['last_online_code', 'drinks_code', 'drugs_code', 'speaks_code']]
x_scale = scale(X)
x_train, x_test, y_train, y_test = train_test_split(x_scale, y, test_size = 0.2, random_state=4949)
scores = []
#loop through ks to check the regressor's performance
for i in range(1,101):
    regressor = KNeighborsRegressor(n_neighbors=i, weights="distance")
    regressor.fit(x_train, y_train)
    scores.append(regressor.score(x_test, y_test))
#plot chart of R2 against k
k_list = list(range(1, 101))
plt.plot(k_list, scores)
plt.xlabel("k")
plt.title("Age Regressor Scores")
plt.show()
#plot chart of actual and predicted ages for what seems the best k
regressor = KNeighborsRegressor(n_neighbors=30, weights="distance")
regressor.fit(x_train, y_train)
y_predict = regressor.predict(x_test)
print(regressor.score(x_test, y_test))
plt.scatter(y_test, y_predict, alpha=0.4)
plt.xlabel('Actual Age')
plt.ylabel('Predicted Age')
plt.title('Actual Vs Predicted Age')
plt.show()
