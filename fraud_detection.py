import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn. neighbors import LocalOutlierFactor
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix,classification_report


df=pd.read_csv(r'C:\Users\Abraham\Desktop\DOINGS\creditcard.csv')

print("\ndata information")
print(df.info())

print("\ndata headings")
print(df.head())

print("\nmissing values")
print(df.isnull().sum())

#ANOMALLY DICTECTION


# Select features for anomaly detection

#DROPING THE TIME COLMN
df=df.drop(columns=['Time'])



#normalizing the amount
scaler = StandardScaler() 
df['Amount']= scaler.fit_transform(df[['Amount']])

#Separating feature X and target y

x= df. drop(columns=['Class'])
y= df['Class']

#training isolation forest

Iso_forest = IsolationForest(contamination= 0.01, random_state=42)
df['anomaly_score']=Iso_forest.fit_predict(x)

#Mark anomalies (-1 is anomaly, 1 is normal)

df['anomaly']= df['anomaly_score'].apply(lambda x: 1 if x == -1 else 0)

#showing the anomaly dectected

fraud_cases = df[df['anomaly']==1]

print(f"Dectected {len(fraud_cases)}potential fraud cases")
print (fraud_cases.head())

#local outlier factor

#apply LOF

lof = LocalOutlierFactor(n_neighbors=20, contamination=0.01)

df['lof_score'] = lof.fit_predict(x)

#marking anomalies  (-1 is anomaly, 1 is normal)
df['lof_anomaly'] =df['lof_score'].apply(lambda x: 1 if x == -1 else 0)

#show anomalies dectected
lof_fraud_cases = df['lof_anomaly']==1
print (f"LOF dectected {len(lof_fraud_cases)}potiential fraud cases")
print(lof_fraud_cases. head())

#comparing dectected anomalies with actual fraud cases (class = 1)

true_fraud = df[df['Class']==1]
detected_fraud = df[(df['anomaly']==1)|(df['lof_anomaly']==1)]


#performance check

true_positives = len(detected_fraud[detected_fraud['Class']==1])

false_positives = len(detected_fraud[detected_fraud['Class']==0])

false_negatives = len(true_fraud) - true_positives

print(f"true positives: {true_positives}")
print(f"false positives: {false_positives}")
print(f"false Negatives: {false_negatives}")

#regression model

#define feature x and target variable y

x=df.drop(columns=['Class']) #this features all colomns except class

y=df['Class'] #target: 'Class' (0= normal, 1 = fraud)


#training and testing state (training 80% and testing 20%) set

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size= 0.2, random_state= 42, stratify= y)


model = LogisticRegression()
model.fit(x_train, y_train)

#making prediction on the test set

y_prediction = model.predict(x_test)

print(y_prediction)


#model evaluation 

#calculate accuracy

accuracy = accuracy_score(y_test, y_prediction)

print(f"accuracy: {accuracy: 4f}")

#confusion matrix 

conf_matrix = confusion_matrix(y_test, y_prediction)
print("cofussion matrix:\n", conf_matrix)


#classification report

print("classification report:\n", classification_report(y_test, y_prediction))












