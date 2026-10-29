import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 

## Naive iris
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# load the data 
iris=load_iris()
x=iris.data
y=iris.target

# Data splitting
x_train, x_test, y_train, y_test=train_test_split(x, y, test_size=0.2, random_state=42)

# Initializing and training the gaussian naive bayes classifier
model=GaussianNB()
model.fit(x_train,y_train)

# predicting the test data 
y_pred=model.predict(x_test)

# evaluating the model performance
print(f"Accuracy: {accuracy_score(y_test,y_pred)*100:.2f}%")
print("\nClassification report:\n", classification_report(y_test,y_pred, target_names=iris.target_names))

# Matrix table-- markdown
| matrix | Predicted Positive | Predicted negative |
|---|---|---|
| Actual Positive | True positive(TP) | False negative(FN) |
| Actal Negative | False Positive(FP) | True negative(TN) |

# With imbalanced data 
df=pd.read_csv(r"creditcard.csv")
df.head()

x=df.drop('Class',axis=1)
y=df['Class']

x_train, x_test, y_train, y_test=train_test_split(x, y, test_size=0.2, random_state=42)

model=GaussianNB()
model.fit(x_train,y_train)

y_pred=model.predict(x_test)

# Accuracy score and classification report 
print(f"Accuracy: {accuracy_score(y_test,y_pred)*100:.2f}%")
print("\nClassification report:\n", classification_report(y_test,y_pred))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
cm

## NAIVE BAYES NEWSGROUP

from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# loading the data 
newsgroups = fetch_20newsgroups(
    subset='all')

x,y=newsgroups.data, newsgroups.target

# train test split
xtrain, xtest, ytrain, ytest= train_test_split(x,y, test_size=0.2, random_state=42)

# initializing the tfidVectorizer 
vectorizer = TfidfVectorizer(stop_words="english", max_df=0.5)

# Fit & transform the training data and transform the test data
x_train_tfdif = vectorizer.fit_transform(xtrain)
x_test_tfdif = vectorizer.transform(xtest)

# Initializing the Multinominal naive bayes classifier 
nb_classifier=MultinomialNB()

# train the classifier on the training data 
nb_classifier.fit(x_train_tfdif, ytrain)

# predict the label for test data 
y_pred= nb_classifier.predict(x_test_tfdif)

# Calculating accuracy
accuracy= accuracy_score(ytest, y_pred)

# classifier report 
class_report=classification_report(ytest, y_pred, target_names=newsgroups.target_names)

print(f"Accuracy: {accuracy*100:.2f}%\n")
print(f"Classification Report:\n{class_report}")

## NAIVE BAYES SENTIMENT ANALYSIS 

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

data= { 
    'text':[ 
        "I absolutely love this product, it works perfectly!", 
        "This is the worst purchase I have ever made.",
        "It's okay, nothing special but does the job.",
        "Fantastic customer service and beautiful design.",
        "Horrible quality, it broke on the first day.",
        "I am highly satisfied with the fast shipping.",
        "Waste of money. Do not buy this product.",
        "Pretty good value for the price, I recommend it.", 
        "item arrived late and the product was damaged, very disappointed." 
    ],
    'sentiment': ['positive','negative','neutral', 'positive', 'negative', 'positive', 'negative', 'positive', 'negative'] 
}

df = pd.DataFrame(data)
df


# Split the dataset into features (x) and target variable (y)

X = df['text']
y = df['sentiment']

# Split into training and testing sets 
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)

# 3 text vactorization (TF-IDf Fetaures Extraction)
vectorizer = TfidfVectorizer(stop_words='english', lowercase=True)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# 4 Initialize and train Classifiers 
lr_model = LogisticRegression()
lr_model.fit(X_train_vec, y_train)

nb_model = MultinomialNB() 
nb_model.fit(X_train_vec, y_train)

# 5 Evaluate and compare models 
lr_preds = lr_model.predict(X_test_vec)
nb_preds = nb_model.predict(X_test_vec)

print('---- Model accuracy comparison ----')
print(f'Logistic regression accuracy: {accuracy_score(y_test, lr_preds) *100:.2f} %')
print(f'Multinomial Naive bayes accuracy: {accuracy_score(y_test, nb_preds) *100:.2f} %\n')

# 6 Predict sentiment on new custom texts 
def predict_sentiment(new_texts): 
    new_texts_vec = vectorizer.transform(new_texts)

    lr_predictions = lr_model.predict(new_texts_vec)
    nb_predictions = nb_model.predict(new_texts_vec)

    print('---- Testing custom Input ----')
    for text, lr_pred, nb_pred in zip(new_texts, lr_predictions, nb_predictions): 
        print(f"Text: '{text}'")
        print(f"    -> LR Prediction: **{lr_pred.upper()}**")
        print(f"    -> NB Prediction: **{nb_pred.upper()}**\n")

custom_samples = [ "The iterm arrived broken and late.","It works incredibly well, Love it!", "Not worht the money", "Looks good"]
predict_sentiment(custom_samples)

custom_samples_1 = ["This is the best purchase", "My money is wasted with this purchase", "Amazing Purchase","Can purchase","Would purchase again","It is okay"]
predict_sentiment(custom_samples_1)

## XBOOST 
pip install xgboost
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from xgboost import XGBClassifier, plot_importance

# Loading the dataset
iris=load_iris()
x=iris.data
y=iris.target
feature_names=iris.feature_names
target_name=iris.target_names

# Split the dataset into training and testing
x_train, x_test, y_train, y_test= train_test_split(x,y, test_size=0.2, random_state=42)

# Initializing and training xg boost classifier 
model= XGBClassifier(
    objective="multi:softprob",
    num_class=3,
    random_state=42,
    eval_metric="mlogloss"
)
model.fit(x_train,y_train)

# Predict the test data 
y_pred=model.predict(x_test)

# print standard evaluation test metrics
print(f"XGBoost accuracy: {accuracy_score(y_test, y_pred)*100:.2f}%\n")
print(
    "Classifiction report:\n",
    classification_report(y_test,y_pred, target_names=target_name),
)

# Confusion matrix 
cm=confusion_matrix(y_test,y_pred)
cm

# Plotting classification matrix 
plt.figure(figsize=(6,4.5))
sns.heatmap(
    cm,
    annot= True,
    fmt="d",
    cmap="Blues",
    xticklabels=target_name,
    yticklabels=target_name,
    cbar=False,
    annot_kws={"size":14, "weight": "bold"},
)
plt.title("xgboost classification confution matrix")
plt.xlabel("predicted species group")
plt.ylabel("actual species group")
plt.tight_layout()
plt.show()

# XGboost feature Importance 
plt.figure(figsize=(8,4))
plot_importance(model, importance_type="weight", grid=False, color="skyblue")
plt.title("XGBoost feature importance - iris dataset",fontsize=14, pad=15)
plt.tight_layout()
plt.show()

## PIPELINE AND GRIDSEARCH
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix

## Load the dataset
data = load_iris() 
X,y = data.data, data.target 
labels = data.target_names

## Investigate a pipe line consisting of Standardscaler, PCA and KneighborClassifier 
pipeline = Pipeline([
    ('scaler', StandardScaler()),      # Step1 - standardize features
    ('PCA', PCA(n_components=2),),     # Step2 - Reduce dimensions to 2 using pCA (optional)
    ('knn', KNeighborsClassifier(n_neighbors=5,))  # Step3 - K-nearest neighbour classifier
])


# Split the data into training and test set 

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42, stratify=y)
---> In pipeline no matter what stratify will be y (by default)

pipeline.fit(X_train, y_train)

# Measure the pipeline accuracy on the test data 
test_score = pipeline.score(X_test, y_test)
print(f'pipeline accuracy\n-----------------\ntest score: {test_score:.3f}')

# Get the model predictions
y_pred = pipeline.predict(X_test)

# generate and plot the confusion matrix for the KNN model 

# generate confusion matrix 
conf_matrix = confusion_matrix(y_test, y_pred)

# Create a plot for the confusion matrix 
plt.figure() 
sns.heatmap(conf_matrix, annot = True, cmap='Blues', fmt = 'd',
           xticklabels=labels, yticklabels=labels)

# set the title and lables 

plt.title('Classification pipeline confusion matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')

plt.show()

# pipeline without any specific parameters 
pipeline=Pipeline([
    ("scaler",StandardScaler()),
    ("pca",PCA()),
    ("knn",KNeighborsClassifier())
])

# Define a model parameter grid to search over 
param_grid = {'pca__n_components': [2,3], 
              'knn__n_neighbors': [3,5,7]}

# Choose a cross validation method 
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Determine the best parameters  
best_model = GridSearchCV(estimator=pipeline_a, 
                         param_grid=param_grid, 
                         cv=cv, 
                         scoring='accuracy', 
                         verbose=2
                         )

# Fit the best gridsearchcv model to the training data 
best_model.fit(X_train, y_train)

# Evaluating the accuracy score of the best model on test set 
test_score = best_model.score(X_test, y_test)
print(f'{test_score*100:.3f}%')

# Display the best parameters 
best_model.best_params_

# Plot the confusion matrix for the prediction on the test set 
y_pred = best_model.predict(X_test)

# generate confusion matrix 
conf_matrix_a = confusion_matrix(y_test, y_pred)

# Create a plot for the confusion matrix 
plt.figure() 
sns.heatmap(conf_matrix_a, annot = True, cmap='Blues', fmt = 'd',
           xticklabels=labels, yticklabels=labels)

# set the title and lables 

plt.title('Classification Accurate pipeline confusion matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')

plt.show()

## RANDOM FOREST 
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# load the data 
data=load_iris()
x=data.data
y=data.target

# Display features and target names
print("Features:", iris.feature_names)
print("Target names:", iris.target_names)

# Create a DataFrame for easier exploration
df = pd.DataFrame(X, columns=iris.feature_names)
df['target'] = iris.target

# Display the first few rows of the dataset
print(df.head())

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the Random Forest model
rf_model = RandomForestClassifier()
rf_model.fit(X_train, y_train)

# Predict on the test set
y_pred = rf_model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print("Accuracy:", accuracy)
print("Classification Report:\n", report)
print("Confusion Matrix:\n", conf_matrix)

# Visualize the dataset
import matplotlib.pyplot as plt
import seaborn as sns
sns.pairplot(df, hue='target', palette='viridis')
plt.show()

