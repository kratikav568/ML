pip install numpy pandas kagglehub skimpy shap matplotlib xgboost yellowbrick scikit-learn

import numpy as np
import pandas as pd
import kagglehub
import os
import skimpy
import shap
import matplotlib.pyplot as plt
import seaborn as sns 
import warnings

from xgboost import XGBClassifier, plot_importance
from skimpy import skim 
from sklearn import linear_model
from yellowbrick.model_selection import LearningCurve
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier
from sklearn.inspection import permutation_importance
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    log_loss, mean_absolute_error, mean_squared_error,
    r2_score, accuracy_score, classification_report,
    confusion_matrix, silhouette_samples, silhouette_score
)
from sklearn.preprocessing import (
    normalize, StandardScaler, OneHotEncoder,
    OrdinalEncoder, FunctionTransformer
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.decomposition import PCA

# IRIS DATASET
from sklearn.datasets import load_iris

# NEWS20GROUP DATASET
from sklearn.datasets import fetch_20newsgroups

# DIABETES DATASET
from sklearn.datasets import load_diabetes

warnings.filterwarnings('ignore')


=====
# DATASET- iris dataset  
data = load_iris() 
X,y = data.data, data.target 
labels = data.target_names

## Investigate a pipe line consisting of Standardscaler, PCA and KneighborClassifier 
pipeline = Pipeline([
    ('scaler', StandardScaler()),      # Step1 - standardize features
    ('PCA', PCA(n_components=2),),     # Step2 - Reduce dimensions to 2 using pCA (optional)
    ('knn', KNeighborsClassifier(n_neighbors=5,))  # Step3 - K-nearest neighbour classifier
])

# Exercise 1 - Split the data into training and test set 
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42, stratify=y)
# In pipeline no matter what stratify will be y (by default)
pipeline.fit(X_train, y_train)

# Measure the pipeline accuracy on the test data 
test_score = pipeline.score(X_test, y_test)
print(f'{test_score:.3f}')

## Get the model predictions
y_pred = pipeline.predict(X_test)

# Exercise 2 - generate and plot the confusion matrix for the KNN model 
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

# A Actual way to create a pipeline 
pipeline_a = Pipeline(
    [('scaler', StandardScaler()), 
     ('pca', PCA()),
     ('knn', KNeighborsClassifier())
        
    ]
)
# Define a model parameter grid to search over 
param_grid = {'pca__n_components': [2,3], 
              'knn__n_neighbors': [3,5,7]} 
## Choose a cross validation method 
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Exercise 4 - Determine the best parameters  
best_model = GridSearchCV(estimator=pipeline_a, 
                         param_grid=param_grid, 
                         cv=cv, 
                         scoring='accuracy', 
                         verbose=2
                         )
# Fit the best gridsearchcv model to the training data 
best_model.fit(X_train, y_train)

# Exercise 5 - 
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
=====

# LOGISTIC REGRESSION - DATASET - Obesity 
obesity_lv = pd.read_csv("C:/Users/pc/Desktop/MLM/Obesity_level_prediction_dataset.csv")
obesity_lv.head()
skim(obesity_lv)
# Data eploration 
# Visualize the distribution of the target variable to understand the class balance 
# Distribution of target variable 
sns.countplot(y='NObeyesdad', data=obesity_lv, palette="Set1")
plt.title("Distribution of obesity levels")
plt.show()

# PROCESSING THE DATA 
# Scale the numerical features to standardize thier range for btter model performance 
# Standardizing countinous numerical features 
continous_columns = obesity_lv.select_dtypes(include=['float64']).columns.tolist() 

scaler = StandardScaler() 
scaled_features = scaler.fit_transform(obesity_lv[continous_columns])

# Converting to dataframe
scaled_df = pd.DataFrame(scaled_features, columns=scaler.get_feature_names_out(continous_columns))

scaled_data = pd.concat([obesity_lv.drop(columns=continous_columns), scaled_df],axis =1)
scaled_data.head()

# Convert categorical variables into numerical format using one - hot - encoding 
# Identifying categorical columns 
categorical_columns = scaled_data.select_dtypes(include=['object']).columns.tolist() 
categorical_columns.remove('NObeyesdad') 

encoder = OneHotEncoder(sparse_output=False, drop='first')
encoded_features = encoder.fit_transform(scaled_data[categorical_columns])
encoded_df = pd.DataFrame(encoded_features, columns = encoder.get_feature_names_out(categorical_columns))
prepped_data = pd.concat([scaled_data.drop(columns=categorical_columns), encoded_df], axis=1)
prepped_data.head()

# ENCODING The TARGET VARIABLE 

prepped_data['NObeyesdad'] = prepped_data['NObeyesdad'].astype('category').cat.codes 
prepped_data.head()

# Seperate the input and target data  
# Preparing final datset 
X = prepped_data.drop('NObeyesdad', axis=1)
y = prepped_data['NObeyesdad']

# Splitting the data set
X_train, X_test, y_train,y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
model_ova = OneVsRestClassifier(
    LogisticRegression(max_iter=1000)
)

# Fit the model
model_ova.fit(X_train, y_train)

# Predict test data
y_pred_ova = model_ova.predict(X_test)
# Evaluating metrics for OvA 
print('One-vs-all (OvA) strategy')
print(f'Accuracy: {np.round(100*accuracy_score(y_test, y_pred_ova), 2)}%')
# Training logistic regresion model usign one vs one 
model_ovo = OneVsOneClassifier(LogisticRegression(max_iter=1000))
model_ovo.fit(X_train, y_train)
# Evaluate the accuracy of the trained model as a measure of its performance on unseen testing data 
# predictions 
y_pred_ovo = model_ovo.predict(X_test)
# Evaluation metrics for OVO 
print('One Vs One (OVO strategy)')
print(f'Accuracy: {np.round(100*accuracy_score(y_test, y_pred_ovo), 2)}%')
====
====
# DATASET - DIABETES - KNN REGRESSION MODEL
sns.set_style('whitegrid')
diabetes = load_diabetes(scaled=False)
X= pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
y= pd.Series(diabetes.target, name="disease_progression")
X["sex"] = X["sex"].map({1:0,2:1})
df = pd.concat([X,y], axis = 1)

print("Dataset shape:", df.shape)
print("Missing Values:",int(df.isnull().sum().sum()))
df.head(3)

# Sumamry stats 
df.describe().T.style.background_gradient(cmap='Blues')
# Distribution of the target variable  
plt.figure(figsize=(8,5))
sns.histplot(y, bins=30, kde=True)
plt.title('Distribution of disease progression')
plt.xlabel("Disease progression")
plt.ylabel('Frequency')
plt.show()

# Heatmap - correlation as the values here are numerical  
corr= df.corr() 
mask = np.triu(np.ones_like(corr, dtype=bool))

plt.figure(figsize=(10,8))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="viridis", mask=mask)
plt.title("Correlation heatmap of Diabetes Dataset")
plt.show()

## Scatter plot to understand the relationship between each variable  
features = list(X.columns)

fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(10,8), sharey=True)
axes = axes.flatten() 

for i, col in enumerate(features): 
    sns.scatterplot(data=df, x=col, y="disease_progression", ax= axes[i], 
                   alpha=0.6, edgecolor=None)
    sns.regplot(data=df, x=col, y="disease_progression", ax=axes[i], 
               scatter=False, color='red')
    axes[i].set_title(f"{col} vs progression", fontsize=11)
    axes[i].set_xlabel(col) 

axes[0].set_ylabel("disease_progression")
axes[5].set_ylabel("disease_progression")

fig.suptitle("All feature vs disease progression", y =1.02, fontsize = 16)
plt.tight_layout()
plt.show()
# Data splitting 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)

# Creating a pipeline 
CONTINOUS = ["age", "bmi", "bp", "s1", "s2", "s3", "s4", "s5", "s6"]
BINARY = ["sex"]

def make_pipeline(features, **knn_params): 
    num = [f for f in CONTINOUS if f in features]
    binary = [f for f in BINARY if f in features]
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num), 
            ("bin", "passthrough", binary), 
        ],
        remainder="drop",
    )
    return Pipeline([
        ("preprocessor", preprocessor),
        ("knn", KNeighborsRegressor(**knn_params)),
        
    ])
    
baseline_pipeline = make_pipeline(list(X.columns))
baseline_pipeline

# Training the baseline model 
baseline_pipeline.fit(X_train, y_train)

## Evaluation score card 
def evaluate_model(model, X_train, y_train, X_test, y_test, label):
    rows = []
    for part, (A, b) in {"Train": (X_train, y_train), "Test": (X_test, y_test)}.items():
        pred = model.predict(A)
        rows.append({
            "Model": label,
            "Data": part,
            "R²": r2_score(b, pred),
            "RMSE": np.sqrt(mean_squared_error(b, pred)),
            "MAE": mean_absolute_error(b, pred),
        })
    return pd.DataFrame(rows)
 baseline_scores = evaluate_model(baseline_pipeline, X_train, y_train, X_test, y_test,
                                 "Baseline KNN (k=5, Euclidean)")
baseline_scores.round(4)

- Since target variable's std dev is 77...and 400 obeservation , so even the smallest difference makes a huge impact   hence we need to tune it 

## Hyperparameter tuning  
param_grid = {
    "knn__n_neighbors": [5,7,9,11,13,15,21,25,31,41,51], 
    "knn__weights": ["uniform","distance"], 
    "knn__p": [1,2],
                
}

grid_search = GridSearchCV(
    estimator = make_pipeline(list(X.columns)), 
    param_grid=param_grid, 
    scoring = "r2", 
    cv=5, 
    n_jobs=1,
)
grid_search.fit(X_train, y_train)

best_idx = grid_search.best_index_
print("Best parameters:", grid_search.best_params_)
print(f"Mean CV R²: {grid_search.best_score_:.4f} "
      f"(fold std {grid_search.cv_results_['std_test_score'][best_idx]:.4f})")
cv = pd.DataFrame(grid_search.cv_results_)
cv["config"] = cv["param_knn__weights"].astype(str) + ",P=" + cv["param_knn__p"].astype(str)

plt.figure(figsize=(9,5))
sns.lineplot(data=cv, x="param_knn__n_neighbors", y = "mean_test_score", 
            hue = "config", markers="o")

plt.xlabel('Number of neighbor (k)')
plt.ylabel('Mean cross- validation R2')
plt.title('Cross-validation r2 across the tuning grid')
plt.legend(title="Weights, p")
plt.show()

# Evaluate on the test set 
tuned_scores = evaluate_model(grid_search.best_estimator_, X_train, y_train, X_test, y_test, 
                             "Tunes KNN")
pd.concat([baseline_scores, tuned_scores]).round(4).reset_index(drop=True)
perm= permutation_importance(
    grid_search.best_estimator_, X_test, y_test, 
    n_repeats=30, random_state=42, scoring='r2',
) 
perm_importance = (pd.DataFrame({
    "feature":X.columns, 
    "importance_mean": perm.importances_mean, 
    "importance_std": perm.importances_std,
})
.sort_values("importance_mean", ascending=False))

plt.figure(figsize=(10,6))
plt.barh(perm_importance["feature"], perm_importance["importance_mean"], 
        xerr= perm_importance["importance_std"])
plt.xlabel("Permutation importance")
plt.show()

# Feature ablation 
best_knn_params = {k.replace("knn__",""): v for k, v in grid_search.best_params_.items()}

base_model = make_pipeline(list(X.columns), **best_knn_params).fit(X_train,y_train)
base_r2 = r2_score(y_test, base_model.predict(X_test))

ablation_rows =[]
for col in X.columns: 
    kept = [f for f in X.columns if f !=col]
    model = make_pipeline(kept, **best_knn_params).fit(X_train[kept], y_train)
    r2 = r2_score(y_test, model.predict(X_test[kept]))
    ablation_rows.append({"feature": col,"r2_drop": base_r2 -r2}) 


ablation = pd.DataFrame(ablation_rows).sort_values(
    "r2_drop", ascending=False
)

plt.figure(figsize=(10, 6))
plt.barh(ablation["feature"], ablation["r2_drop"])
plt.gca().invert_yaxis()
plt.xlabel("Drop in R² after removing the feature")
plt.title("Leave-One-Feature-Out Ablation")
plt.grid(True, axis="x")
plt.tight_layout()
best_pipe = grid_search.best_estimator_

def predict_fn(arr):
    """Prediction function that accepts a plain array and restores the column names.""" 
    return best_pipe.predict(pd.DataFrame(arr, columns=X.columns))

background = shap.kmeans(X_train.to_numpy(), 30)
explainer = shap.KernelExplainer(predict_fn, background)
shap_values = explainer.shap_values(X_test.to_numpy(), silent = True)
# Summary plot 
shap.summary_plot(shap_values, X_test)
# Shap Dependence plots 
fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(12, 8))
axes = axes.flatten()

for i, col in enumerate(X.columns):
    shap.dependence_plot(
        ind=i,
        shap_values=shap_values,
        features=X_test.to_numpy(),
        feature_names=list(X.columns),
        interaction_index="auto",
        ax=axes[i],
        show=False,
        dot_size=30,
        alpha=0.8,
    )
    axes[i].set_title(col, fontsize=11)

best_k = best_knn_params["n_neighbors"]
plt.suptitle(
    f"SHAP Dependence Plots - Tuned KNN (k={best_k})",
    y=1.02,
    fontsize=16
)
plt.tight_layout()
plt.show()
====
====
# DATASET - FuelConsumptionCo2
feul_con = pd.read_csv("C:/Users/pc/Desktop/FuelConsumptionCo2.csv")
feul_con.head()
df= pd.DataFrame(feul_con)
df.sample(5)
# Selecting features
cdf = df[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB','CO2EMISSIONS']]
cdf.sample(9)
# Visualize features 
# creating a histgram for each of the feature 
cdf.hist() 
plt.show()
plt.scatter(cdf.FUELCONSUMPTION_COMB, cdf.CO2EMISSIONS, color = 'blue')
plt.xlabel('Fuel consumption_comb')
plt.ylabel('EMISSION')
plt.show()
plt.scatter(cdf.ENGINESIZE, cdf.CO2EMISSIONS, color='blue')
plt.xlabel('engine size')
plt.ylabel('Emission')
plt.xlim(0,27)
plt.show()
# practic exercise 
plt.scatter(cdf.CYLINDERS, cdf.CO2EMISSIONS, color='orange')
plt.xlabel('Cylinder size')
plt.ylabel('Emission')
plt.show() 

# Extract the input feature and labels from the dataset 
x= cdf.ENGINESIZE.to_numpy() 
y = cdf.CO2EMISSIONS.to_numpy()
print(x)
# create train and test the dataset 
# now you want to randomly split ur data into train and test sets, using 80% of the dataset to train and 20% for the testing. 
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.2, random_state=42)
type(x_train), np.shape(x_train), np.shape(x_train)

#Creating a model object 
regressor = linear_model.LinearRegression() 

# train ther model 
# x_train is a 1D array but sklearn expect a 3D array as input for the training data so we need to reshape it. we can let it infer the number of observation using '-1'
regressor.fit(x_train.reshape(-1,1), y_train)

# print the coefficient 
print('coefficients: ', regressor.coef_[0]) # exptracting from 1 by 1 array
print('intercept: ', regressor.intercept_)
# visualizing model outputs 
plt.scatter(x_train, y_train, color= 'black')
plt.plot(x_train, regressor.coef_*x_train + regressor.intercept_, ':r')
plt.xlabel('Engine size')
plt.ylabel('Emission')
# model evaluation 
# use the predict method to make test predictions 
y_pred = regressor.predict(x_test.reshape(-1,1))

# evaluation
print('Mean absolute error: %.2f' % mean_absolute_error(y_test, y_pred))
print('Mean squared error: %.2f' % mean_squared_error(y_test, y_pred))
print('Root mean squared error: %.2f' % np.sqrt(mean_squared_error(y_test, y_pred)))
print('R2-score: %.2f' % r2_score(y_test, y_pred))
====
====
# DATASET - customer raw
customers = pd.read_csv("C:\\Users\\pc\\Desktop\\MLM\\customers_raw.csv")
customers.sample(5)
customers.head()
print("\nShape:", customers.shape)
purchases = pd.read_csv("C:\\Users\\pc\\Desktop\\MLM\\purchases.csv")
purchases.head()
print("\nShape:", purchases.shape)

print("\nData Types")
print(customers.dtypes)
print("\nInformation")
customers.info()

# Missing values 
print("\nMissing Values")
print(customers.isnull().sum())
print("\nMissing Values")
print(purchases.isnull().sum())

for col in customers.columns:
    print(f"{col}: {customers[col].nunique()}")
for col in customers.select_dtypes(include='object'):
    print("\n", "="*40)
    print(col)
    print(customers[col].value_counts(dropna=False))

# Outlier detection 
numeric_cols = customers.select_dtypes(include=np.number).columns

for col in numeric_cols:
    Q1 = customers[col].quantile(0.25)
    Q3 = customers[col].quantile(0.75)
    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    outliers = customers[(customers[col] < lower) | (customers[col] > upper)]

    print(f"\n{col}")
    print("Outliers:", len(outliers))

# plots 
for col in numeric_cols:
    plt.figure(figsize=(5,3))
    plt.boxplot(customers[col].dropna())
    plt.title(col)
    plt.show()

# Histogram 
customers[numeric_cols].hist(figsize=(8,5))
plt.tight_layout()
plt.show()

# Removing duplicate customers based on CustomerId
customers = customers.drop_duplicates(subset="CustomerID", keep="first")
# Remove extra spaces and convert to title case
categorical_cols = ["Name", "City", "Gender"]

for col in categorical_cols:
    if col in customers.columns:
        customers[col] = (
            customers[col]
            .astype(str)
            .str.strip()
            .str.title()
        )

#1b - standardize inconsistent categorical text 
customers["City"] = customers["City"].str.strip().str.title()
print("(b) Standardized 'City' text casing:")
print(customers["City"].unique())

# Handling missing values 
# Fill missing names with "Unknown"
customers["Name"] = customers["Name"].fillna("Unknown")

# Fill missing Age with median
customers["Age"] = customers["Age"].fillna(customers["Age"].median())

# Fill missing Salary with median
customers["Salary"] = customers["Salary"].fillna(customers["Salary"].median())
print(customers.isnull().sum())
print(customers.head())

df_integrated=pd.merge(customers, purchases, on="CustomerID", how="left")
 
df_integrated["TotalPurchases"]=df_integrated["TotalPurchases"].fillna(0)
df_integrated["MemberSince"]=df_integrated["MemberSince"].fillna(df_integrated["MemberSince"].median())
 
print("\nInegrated dataset (customer+purchase info merged on Customer ID):")
print(df_integrated)

# data transformation 
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder
df_t=df_integrated.copy()
#3a - Normalization (Min-max scaling :0,1)- useful for diastance based models 
minmax=MinMaxScaler()
df_t[["Age_MinMax", "Salary_MinMax"]]=minmax.fit_transform(df_t[["Age", "Salary"]])

#3b - Standardization 
zscore=StandardScaler()
df_t[["Age_Z", "Salary_Z"]]=zscore.fit_transform(df_t[["Age","Salary"]])
 
#3c - Encoding categorical variables 
le=LabelEncoder()
df_t["Gender_Encoded"]=le.fit_transform(df_t["Gender"])
df_t=pd.get_dummies(df_t, columns=["City"],prefix="City")
 
#3d - discretization/binning - convert continuous age into categorical buckets 
bins=[0,25,35,50,100]
labels=["Young", "Adult","MidAge", "Senior"]
df_t["Age_Group"]=pd.cut(df_integrated["Age"], bins=bins, labels=labels)
 
print("Transformed dataset (scaled, encoded and binned):")
show_cols=["CustomerID", "Age", "Age_MinMax", "Age_Z", "Age_Group", "Salary", "Salary_MinMax", "Salary_Z", "Gender", "Gender_Encoded"]
print(df_t[show_cols])


print("\nMerged Shape:", merged.shape)
print("\nMerged Missing Values")
print(merged.isnull().sum())

# Correlations 
numeric = merged.select_dtypes(include=np.number)

print("\nCorrelation Matrix")
print(numeric.corr())

# Visualization 
# Gender Distribution
if "Gender" in merged.columns:
    plt.figure(figsize=(5,4))
    merged["Gender"].value_counts(dropna=False).plot(kind="bar")
    plt.title("Gender Distribution")
    plt.ylabel("Count")
    plt.show()

# City Distribution
if "City" in merged.columns:
    plt.figure(figsize=(7,4))
    merged["City"].value_counts(dropna=False).plot(kind="bar")
    plt.title("City Distribution")
    plt.ylabel("Count")
    plt.xticks(rotation=45)
    plt.show()

# Salary Distribution
if "Salary" in merged.columns:
    plt.figure(figsize=(6,4))
    plt.hist(merged["Salary"].dropna(), bins=10)
    plt.title("Salary Distribution")
    plt.xlabel("Salary")
    plt.ylabel("Frequency")
    plt.show()

# Age vs Salary
if {"Age","Salary"}.issubset(merged.columns):
    plt.figure(figsize=(6,4))
    plt.scatter(merged["Age"], merged["Salary"])
    plt.title("Age vs Salary")
    plt.xlabel("Age")
    plt.ylabel("Salary")
    plt.show()

print("\nEDA Completed Successfully.")


merged = pd.get_dummies(
    merged,
    columns=['City'],
    drop_first=True
)

merged.head()
# Calculating cutomer loyalty scores 

merged['MemberSince'] = pd.to_datetime(merged['MemberSince'])

present = pd.Timestamp.today()

merged['Member_years'] = (
    present - merged['MemberSince']
).dt.days/365.25
# Calculating purchase point 
def purchase_points(x):
    if x >=20:
        return 3   
    elif x>=15:
        return 2    
    elif x>=5: 
        return 1 
    else: 
        return 0 

merged['Purchase_points'] = (
    merged['TotalPurchases'].apply(purchase_points)
)
# Calculating membership point 
def member_points(x):
    if x >=6:
        return 3   
    elif x>=4:
        return 2    
    elif x>=2: 
        return 1 
    else: 
        return 0 

merged['Member_points'] = (
    merged['Member_years'].apply(member_points)
)
# Customer Lifetime Value score 
merged['CLV'] = (
    merged['Purchase_points']+ merged['Member_points']
)
# Calculating Customer Card
def Customer_card(score):
    if score >=9:
        return "Platinum"   
    elif score>=7:
        return "Gold"   
    elif score>=5: 
        return "Silver" 
    else: 
        return "Bronze"

merged['Customer_card'] = (
    merged['CLV'].apply(Customer_card)
)
merged.head()
====
====
# DATASET - churn
c_churn = pd.read_csv("C:/Users/pc/Desktop/MLM/ChurnData.csv")
c_churn.head()
c_churn = c_churn[['tenure','age','address','income','ed','employ','equip','churn']]
c_churn['churn'] = c_churn['churn'].astype('int')
c_churn
- for modeling the input fields X and the target field Y need to be fixed. Since that the target to be predicted 'Churn', the data under this diled will be stored under the variable 'y'. We may use any combo or all of the remainf dields as the inpiut. store these values in teh avriable 'X 
X = np.asarray(c_churn[['tenure','age','address','income','ed','employ','equip']])
X[0:5]
y = np.asarray(c_churn['churn'])
y[0:5]
X_norm = StandardScaler().fit(X).transform(X)
X_norm[:5]

# Splitting the dataset 
X_train, X_test, y_train, y_test = train_test_split(X_norm, y, test_size=0.2, random_state=4)

# Logistic regression 
LR = LogisticRegression().fit(X_train,y_train)
yhat = LR.predict(X_test)
yhat[:10]
yhat_prob = LR.predict_proba(X_test)
yhat_prob[:10]
Coefficients = pd.Series(LR.coef_[0], index= c_churn.columns[:-1])
Coefficients.sort_values().plot(kind='barh')
plt.title('Feature Coefficients in Logistic Regression churn model')
plt.xlabel('coefficient value')
plt.show()
log_loss(y_test, yhat_prob)

# 1. Load dataset (no missing value or outlier treatment applied)
dfc = pd.read_csv("C:/Users/pc/Desktop/MLM/ChurnData.csv")
X = dfc.drop('churn', axis=1)
y = dfc['churn'] 

# 2. 80:20 Train-Test Split with random_state=42 and no stratification
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=None
)
# 3. Select 10 best features using ANOVA F-test (fitted on training data to prevent data leakage)
selector = SelectKBest(score_func=f_classif, k=10)
X_train_best = selector.fit_transform(X_train, y_train)
X_test_best = selector.transform(X_test)
selected_features = X.columns[selector.get_support()].tolist()
print("Top 10 Selected Features:", selected_features)

# 4. Logistic Regression (10 best features)
log_reg = LogisticRegression(max_iter=1000, random_state=42)
log_reg.fit(X_train_best, y_train)
lr_acc = accuracy_score(y_test, log_reg.predict(X_test_best))
print(f"\nLogistic Regression Accuracy: {lr_acc:.4f} ({lr_acc * 100:.2f}%)")

# 5. KNN with K=4 (10 best features)
knn_4 = KNeighborsClassifier(n_neighbors=4)
knn_4.fit(X_train_best, y_train)
knn4_acc = accuracy_score(y_test, knn_4.predict(X_test_best))
print(f"KNN (K=4) Accuracy: {knn4_acc:.4f} ({knn4_acc * 100:.2f}%)")

# 6. Hyperparameter Tuning for KNN (K = 1 to 10)
print("\n--- KNN Hyperparameter Tuning (K=1 to 10) ---")
k_accuracies = {}
for k in range(1, 11):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train_best, y_train)
    acc = accuracy_score(y_test, knn.predict(X_test_best))
    k_accuracies[k] = acc
    print(f"K = {k:2d} | Accuracy: {acc:.4f} ({acc * 100:.2f}%)")

max_acc = max(k_accuracies.values())
best_ks = [k for k, acc in k_accuracies.items() if acc == max_acc]
print(f"\nBest K value(s): {best_ks} with Highest Accuracy: {max_acc:.4f} ({max_acc * 100:.2f}%)")

# Optional: Standardized features comparison (recommended for distance-based KNN)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_best)
X_test_scaled = scaler.transform(X_test_best)

scaled_k_accuracies = {
    k: accuracy_score(y_test, KNeighborsClassifier(n_neighbors=k).fit(X_train_scaled, y_train).predict(X_test_scaled))
    for k in range(1, 11)
}
max_scaled_acc = max(scaled_k_accuracies.values())
best_scaled_ks = [k for k, acc in scaled_k_accuracies.items() if acc == max_scaled_acc]
print(f"\n[With StandardScaler] Best K value(s): {best_scaled_ks} with Highest Accuracy: {max_scaled_acc:.4f} ({max_scaled_acc * 100:.2f}%)")
====
====
# DATASET - yellowtrip  
trip = pd.read_csv("C:/Users/pc/Desktop/MLM/yellow_tripdata.csv")
trip.head()
trip.isnull().sum()
skim(trip)

# Let us plot a correlation matrics of the target variable againsdt teh input variables 
cor_values = trip.corr()['tip_amount']
cor_values.plot(kind='barh', figsize = (10,6))
cor_values = trip.corr()['tip_amount'].drop('tip_amount')
cor_values.plot(kind='barh', figsize = (10,6))

# Extract the labels from the dataframe
y = trip[['tip_amount']].values.astype('float32')

# drop the target variable from the feature matrix 
proc_data = trip.drop(['tip_amount'], axis = 1)

# get the features matrix used for training 
X= proc_data.values 

# Normalize the feature matrix 
X = normalize(X, axis=1, norm='l1', copy=False)
# Now the dataset is ready fro building classification models now train - test set 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
# For reproductive outpit across multple funtion calls , set random state an interger 
dt_reg = DecisionTreeRegressor(criterion='squared_error',
                              max_depth=3,
                              random_state=35)
features = ['trip_distance', 'passenger_count', 'RatecodeID', 'PULocationID', 'DOLocationID', 'payment_type', 'mta_tax', 'tolls_amount', 'improvement_surcharge', 'fare_amount']
trip = trip.dropna(subset=features + ['tip_amount'])

X = trip[features]
y = trip['tip_amount']

dt_reg.fit(X, y)

plt.figure(figsize=(12, 6))
plot_tree(dt_reg, feature_names=features, filled=True)
plt.show()
dt_reg.fit(X_train, y_train)
# run interference using sklearn model
y_pred = dt_reg.predict(X_test)

mse_score = mean_squared_error(y_test, y_pred)
print('MSE score : {0:.3f}'.format(mse_score))

r2_score = dt_reg.score(X_test, y_test)
print('R^2 score : {0:.3f}'.format(r2_score))
dt_reg1 = DecisionTreeRegressor(criterion='squared_error',
                              max_depth=12,
                              random_state=35)

dt_reg1.fit(X_train, y_train)

y_pred1 = dt_reg1.predict(X_test)

mse_score1 = mean_squared_error(y_test, y_pred1)
print('MSE score : {0:.3f}'.format(mse_score1))

r2_score1 = dt_reg1.score(X_test, y_test)
print('R^2 score : {0:.3f}'.format(r2_score1))
====
====
# SENTIMENT ANALYSIS USING NAIVE BAYES 
ratings = pd.read_csv("C:/Users/pc/Downloads/IMDB-Dataset.csv")
dfr = pd.DataFrame(ratings)
dfr.head()
# Split the dataset into features (x) and target variable (y)

X = dfr['review']
y = dfr['sentiment']

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
====
====
# DATASET - fetch_20newsgroups - Naive bayes 
newsgroups = fetch_20newsgroups(
    subset='all')
x,y=newsgroups.data, newsgroups.target
# train test split 
xtrain, xtest, ytrain, ytest= train_test_split(x,y, test_size=0.2, random_state=42)
# Initializing the vectorizer  
vectorizer = TfidfVectorizer(stop_words="english", max_df=0.5)
# Fit and transform  
x_train_tfdif = vectorizer.fit_transform(xtrain)
x_test_tfdif = vectorizer.transform(xtest)
# Multinomial naive bayes 
nb_classifier=MultinomialNB()
# train the classifier on training data 
nb_classifier.fit(x_train_tfdif, ytrain)
# predict the label for test data  
y_pred= nb_classifier.predict(x_test_tfdif)
# accuracy , classifcation report
accuracy= accuracy_score(ytest, y_pred) 

class_report=classification_report(ytest, y_pred, target_names=newsgroups.target_names) 
print(f"Accuracy: {accuracy*100:.2f}%\n")
print(f"Classification Report:\n{class_report}")
====
====
## GUASSIAN NAIVE BAYES WITH IMBALANCE DATA  
dfcc =pd.read_csv(r"C:/Users/pc/Desktop/MLM/creditcard(in) (1).csv")
dfcc.head()

x=df.drop('Class',axis=1)
y=df['Class'] 
x_train, x_test, y_train, y_test=train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)
model=GaussianNB()
model.fit(x_train,y_train)
y_pred=model.predict(x_test)
print(f"Accuracy: {accuracy_score(y_test,y_pred)*100:.2f}%")
print("\nClassification report:\n", classification_report(y_test,y_pred)) 
cm = confusion_matrix(y_test, y_pred)
cm
====
====
# DATASET - IRIS - RANDOM BOOST  
data=load_iris()
x=data.data
y=data.target
x_train, x_test, y_train, y_test= train_test_split(x,y, test_size=0.35, random_state=42) 
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42) 
rf_classifier.fit(x_train,y_train) 
y_pred=rf_classifier.predict(x_test) 
accuracy=accuracy_score(y_test,y_pred)
classification_rep = classification_report(y_test, y_pred)
print(f"accuracy: {accuracy*100:.2f}%")
print("classification report:")
print(classification_rep)
====
====
# DATASET - IRIS - XGBOOST 
iris=load_iris()
x=iris.data
y=iris.target
feature_names=iris.feature_names
target_name=iris.target_names
x_train, x_test, y_train, y_test= train_test_split(x,y, test_size=0.2, random_state=42) 
model= XGBClassifier(
    objective="multi:softprob",
    num_class=3,
    random_state=42,
    eval_metric="mlogloss"
)
model.fit(x_train,y_train) 
y_pred=model.predict(x_test) 
print(f"XGBoost accuracy: {accuracy_score(y_test, y_pred)*100:.2f}%\n")
print(
    "Classifiction report:\n",
    classification_report(y_test,y_pred, target_names=target_name),
) 
cm=confusion_matrix(y_test,y_pred) 
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
plt.figure(figsize=(8,4))
plot_importance(model, importance_type="weight", grid=False, color="skyblue")
plt.title("XGBoost feature importance - iris dataset",fontsize=14, pad=15)
plt.tight_layout()
plt.show()
====
====
# DATASET - PIPELINE AND GRIDSEARCH - IRIS 
data=load_iris()
x,y= data.data, data.target
labels=data.target_names
### pipeline consisting of standard scaler pca and kneighborsclassifiers 
pipeline=Pipeline([
    ("scaler",StandardScaler()), # standardise the features
    ("pca",PCA(n_components=2)), # reduce the dimention to 2 usinng pca
    ("knn",KNeighborsClassifier(n_neighbors=5,))
]) 
### train test split 
xtrain, xtest, ytrain, ytest= train_test_split(x,y, test_size=0.2, random_state=42, stratify=y) 
### fitting the pipeline and accuracy of the pipeline 
pipeline.fit(xtrain,ytrain)

# accuracy
test_score=pipeline.score(xtest,ytest)
print(f"pipeline accuracy\n-----------------\ntest score: {test_score:.3f}")
## prediction 
ypred=pipeline.predict(xtest) 
### confution matrix for knn and plotting 
# generating cconfusion_matrix
conf_matrix=confusion_matrix(ytest, ypred)

# plotting confution matrix
plt.figure()
sns.heatmap(conf_matrix, annot= True, cmap="Blues", fmt="d",
           xticklabels=labels, yticklabels=labels)

# set the title ans labels
plt.title("classification pipeline confusion matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
### pipeline without any specific parameter 
pipeline=Pipeline([
    ("scaler",StandardScaler()),
    ("pca",PCA()),
    ("knn",KNeighborsClassifier())
]) 
### define a model parameter grid to search over 
# kyper parameter search grid for number of pca components and knn neighbors
param_grid={"pca__n_components": [2,3],
           "knn__n_neighbors": [3,5,7]
           } 
### choose a cross validation method 
cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42) 
### best parameters 
best_model=GridSearchCV(estimator=pipeline,
                       param_grid=param_grid,
                       cv=cv,
                       scoring="accuracy",
                       verbose=2)
best_model.fit(xtrain,ytrain) 
### accuracy of the best model 
test_score=best_model.score(xtest,ytest)
print(f"test score: {test_score:.3f}")
## displaying the best parameter 
best_model.best_params_
## prediction of the best model 
ypredb=best_model.predict(xtest) 
### confution matrix for knn and plotting 
# generating cconfusion_matrix
conf_matrix=confusion_matrix(ytest, ypred)

# plotting confution matrix
plt.figure()
sns.heatmap(conf_matrix, annot= True, cmap="Blues", fmt="d",
           xticklabels=labels, yticklabels=labels)

# set the title ans labels
plt.title("classification pipeline confusion matrix best model")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
============================================================================================================
# Data Import and Configuration
# Change these values for your dataset.
DATA_PATH = "your_dataset.csv"       # Example: r"C:/Users/YourName/Downloads/data.csv"
TARGET = "target_column"            # Set the target column name
TASK = "classification"             # Choose: "classification" or "regression"
TEST_SIZE = 0.20
RANDOM_STATE = 42

# CSV import. For Excel use: pd.read_excel("your_dataset.xlsx")
df = pd.read_csv(DATA_PATH)

print("Dataset shape:", df.shape)
display(df.head())

# Data Exploration
df.info()
display(df.describe(include="all").T)
print("\nColumn names:", df.columns.tolist())
print("\nData types:\n", df.dtypes)

## Missing Value Check
missing_summary = pd.DataFrame({
    "Missing Count": df.isna().sum(),
    "Missing %": (df.isna().mean() * 100).round(2)
}).sort_values("Missing %", ascending=False)
display(missing_summary)

## Outlier Check (IQR summary for numeric features)
numeric_columns = df.select_dtypes(include=np.number).columns.tolist()

outlier_rows = []
for col in numeric_columns:
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    count = ((df[col] < lower) | (df[col] > upper)).sum()
    outlier_rows.append([col, lower, upper, count])

display(pd.DataFrame(outlier_rows, columns=["Feature", "IQR Lower", "IQR Upper", "Outlier Count"]))

## Duplicate Value Check
print("Duplicate rows:", df.duplicated().sum())

##Visualising Target and Feature Distributions
# Set TARGET correctly before running this cell.
if TARGET in df.columns:
    plt.figure(figsize=(7, 4))
    if df[TARGET].nunique(dropna=True) <= 20:
        sns.countplot(data=df, x=TARGET)
        plt.xticks(rotation=45)
    else:
        sns.histplot(data=df, x=TARGET, kde=True)
    plt.title(f"Target distribution: {TARGET}")
    plt.tight_layout()
    plt.show()

    feature_columns = [c for c in df.columns if c != TARGET]
    numeric_features = df[feature_columns].select_dtypes(include=np.number).columns.tolist()
    if numeric_features:
        df[numeric_features].hist(figsize=(14, 10), bins=25)
        plt.suptitle("Distribution of numeric features")
        plt.tight_layout()
        plt.show()
else:
    print("Update TARGET to a column present in the dataset.")

    ## Correlation Plot of Features
    corr_data = df.select_dtypes(include=np.number)
if corr_data.shape[1] >= 2:
    plt.figure(figsize=(12, 8))
    sns.heatmap(corr_data.corr(), cmap="coolwarm", center=0, annot=False)
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.show()
else:
    print("At least two numeric columns are needed for a correlation heatmap.")

##ML - WITHOUT PIPELINE 
## Data Preprocessing – Split, Impute, IQR Cap, Remove Duplicates, Encode
# Remove duplicate rows before splitting.
df_clean = df.drop_duplicates().copy()

# Remove rows where the target is missing.
df_clean = df_clean.dropna(subset=[TARGET]).copy()

X = df_clean.drop(columns=[TARGET]).copy()
y = df_clean[TARGET].copy()

# Encode classification target labels if they are text/categories.
label_encoder = None
if TASK == "classification" and (y.dtype == "object" or str(y.dtype).startswith("category")):
    label_encoder = LabelEncoder()
    y = pd.Series(label_encoder.fit_transform(y.astype(str)), index=y.index, name=TARGET)

# Stratify only when classification classes have enough examples.
stratify_y = y if TASK == "classification" and y.value_counts().min() >= 2 else None
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=stratify_y
)

numeric_features = X_train.select_dtypes(include=np.number).columns.tolist()
categorical_features = X_train.select_dtypes(exclude=np.number).columns.tolist()

print("Train shape:", X_train.shape, " Test shape:", X_test.shape)
print("Numeric features:", numeric_features)
print("Categorical features:", categorical_features)

# 1. Missing-value imputation (fit on train, apply to train and test)
num_imputer = SimpleImputer(strategy="median")
cat_imputer = SimpleImputer(strategy="most_frequent")

X_train_num = pd.DataFrame(num_imputer.fit_transform(X_train[numeric_features]),
                           columns=numeric_features, index=X_train.index) if numeric_features else pd.DataFrame(index=X_train.index)
X_test_num = pd.DataFrame(num_imputer.transform(X_test[numeric_features]),
                          columns=numeric_features, index=X_test.index) if numeric_features else pd.DataFrame(index=X_test.index)

X_train_cat = pd.DataFrame(cat_imputer.fit_transform(X_train[categorical_features]),
                           columns=categorical_features, index=X_train.index) if categorical_features else pd.DataFrame(index=X_train.index)
X_test_cat = pd.DataFrame(cat_imputer.transform(X_test[categorical_features]),
                          columns=categorical_features, index=X_test.index) if categorical_features else pd.DataFrame(index=X_test.index)

# 2. IQR method (capping) – bounds calculated from imputed training data only.
iqr_bounds = {}
for col in numeric_features:
    q1 = X_train_num[col].quantile(0.25)
    q3 = X_train_num[col].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    iqr_bounds[col] = (lower, upper)
    X_train_num[col] = X_train_num[col].clip(lower, upper)
    X_test_num[col] = X_test_num[col].clip(lower, upper)

# 3. One-hot encoding of categorical features.
if categorical_features:
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    X_train_cat_encoded = pd.DataFrame(
        encoder.fit_transform(X_train_cat),
        columns=encoder.get_feature_names_out(categorical_features), index=X_train.index
    )
    X_test_cat_encoded = pd.DataFrame(
        encoder.transform(X_test_cat),
        columns=encoder.get_feature_names_out(categorical_features), index=X_test.index
    )
else:
    X_train_cat_encoded = pd.DataFrame(index=X_train.index)
    X_test_cat_encoded = pd.DataFrame(index=X_test.index)

# 4. Combine numeric and encoded categorical features.
X_train_processed = pd.concat([X_train_num, X_train_cat_encoded], axis=1)
X_test_processed = pd.concat([X_test_num, X_test_cat_encoded], axis=1)

# 5. Scale features for distance/margin-based algorithms.
standard_scaler = StandardScaler()
X_train_scaled = pd.DataFrame(standard_scaler.fit_transform(X_train_processed),
                              columns=X_train_processed.columns, index=X_train_processed.index)
X_test_scaled = pd.DataFrame(standard_scaler.transform(X_test_processed),
                             columns=X_test_processed.columns, index=X_test_processed.index)

# Non-negative version for Multinomial Naive Bayes.
minmax_scaler = MinMaxScaler()
X_train_nonnegative = pd.DataFrame(minmax_scaler.fit_transform(X_train_processed),
                                   columns=X_train_processed.columns, index=X_train_processed.index)
X_test_nonnegative = pd.DataFrame(minmax_scaler.transform(X_test_processed),
                                  columns=X_test_processed.columns, index=X_test_processed.index)

print("Processed train shape:", X_train_processed.shape)
print("Processed test shape:", X_test_processed.shape)

## 1. Linear Regression (Regression target only)
# Linear Regression uses unscaled encoded data for ordinary least squares.
linear_model = LinearRegression()
linear_model.fit(X_train_processed, y_train)
y_pred_lr = linear_model.predict(X_test_processed)

print("MAE:", mean_absolute_error(y_test, y_pred_lr))
print("RMSE:", mean_squared_error(y_test, y_pred_lr) ** 0.5)
print("R2:", r2_score(y_test, y_pred_lr))

# Residual diagnostics
residuals = y_test - y_pred_lr
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
sns.scatterplot(x=y_pred_lr, y=residuals, ax=axes[0])
axes[0].axhline(0, color="red", linestyle="--")
axes[0].set_title("Residuals vs Fitted")
sns.histplot(residuals, kde=True, ax=axes[1])
axes[1].set_title("Residual Distribution")
plt.tight_layout()
plt.show()

# Coefficient importance
lr_coef = pd.Series(linear_model.coef_, index=X_train_processed.columns).sort_values(key=np.abs, ascending=False)
display(lr_coef.to_frame("Coefficient"))

## 2. Logistic Regression (Classification target only)
logistic_model = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE)
logistic_grid = GridSearchCV(
    logistic_model,
    {"C": [0.01, 0.1, 1, 10], "solver": ["lbfgs"]},
    cv=5, scoring="f1_weighted", n_jobs=-1
)
logistic_grid.fit(X_train_scaled, y_train)
logistic_best = logistic_grid.best_estimator_
y_pred_lr_clf = logistic_best.predict(X_test_scaled)

print("Best parameters:", logistic_grid.best_params_)
print("Accuracy:", accuracy_score(y_test, y_pred_lr_clf))
print(classification_report(y_test, y_pred_lr_clf, zero_division=0))
ConfusionMatrixDisplay.from_predictions(y_test, y_pred_lr_clf)
plt.title("Logistic Regression – Confusion Matrix")
plt.show()

logistic_coef = pd.Series(logistic_best.coef_[0] if len(logistic_best.coef_) == 1 else np.abs(logistic_best.coef_).mean(axis=0), index=X_train_scaled.columns)
display(logistic_coef.sort_values(ascending=False).head(20).to_frame("Coefficient / Mean absolute coefficient"))

## 3. Decision Tree (Classifier or Regressor)
if TASK == "classification":
    tree_model = DecisionTreeClassifier(random_state=RANDOM_STATE, class_weight="balanced")
    tree_scoring = "f1_weighted"
else:
    tree_model = DecisionTreeRegressor(random_state=RANDOM_STATE)
    tree_scoring = "r2"

tree_grid = GridSearchCV(
    tree_model,
    {"max_depth": [None, 3, 5, 8], "min_samples_split": [2, 5, 10], "min_samples_leaf": [1, 2, 4]},
    cv=5, scoring=tree_scoring, n_jobs=-1
)
tree_grid.fit(X_train_processed, y_train)
tree_best = tree_grid.best_estimator_
y_pred_tree = tree_best.predict(X_test_processed)
print("Best parameters:", tree_grid.best_params_)

if TASK == "classification":
    print("Accuracy:", accuracy_score(y_test, y_pred_tree))
    print(classification_report(y_test, y_pred_tree, zero_division=0))
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred_tree)
    plt.show()
else:
    print("MAE:", mean_absolute_error(y_test, y_pred_tree))
    print("RMSE:", mean_squared_error(y_test, y_pred_tree) ** 0.5)
    print("R2:", r2_score(y_test, y_pred_tree))

pd.Series(tree_best.feature_importances_, index=X_train_processed.columns).sort_values(ascending=False).head(20).plot(kind="barh", figsize=(8, 6))
plt.title("Decision Tree Feature Importance")
plt.gca().invert_yaxis()
plt.show()

## 4. K-Means Clustering (Unsupervised)
# K-Means does not use y. Use scaled features and select k using inertia/silhouette.
inertias = []
silhouette_scores = []
k_values = range(2, 8)
for k in k_values:
    km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
    labels = km.fit_predict(X_train_scaled)
    inertias.append(km.inertia_)
    silhouette_scores.append(silhouette_score(X_train_scaled, labels))

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(list(k_values), inertias, marker="o")
axes[0].set_title("Elbow Method")
axes[0].set_xlabel("Number of clusters (k)")
axes[1].plot(list(k_values), silhouette_scores, marker="o")
axes[1].set_title("Silhouette Score")
axes[1].set_xlabel("Number of clusters (k)")
plt.tight_layout()
plt.show()

# Set the selected cluster count after inspecting the plots.
BEST_K = 3
kmeans_model = KMeans(n_clusters=BEST_K, random_state=RANDOM_STATE, n_init=10)
train_clusters = kmeans_model.fit_predict(X_train_scaled)
test_clusters = kmeans_model.predict(X_test_scaled)
print("Test cluster counts:\n", pd.Series(test_clusters).value_counts().sort_index())
print("Test silhouette score:", silhouette_score(X_test_scaled, test_clusters))

## 5. Support Vector Machine – SVC / SVR
if TASK == "classification":
    svm_model = SVC(class_weight="balanced")
    svm_params = {"C": [0.1, 1, 10], "kernel": ["linear", "rbf"], "gamma": ["scale", "auto"]}
    svm_scoring = "f1_weighted"
else:
    svm_model = SVR()
    svm_params = {"C": [0.1, 1, 10], "kernel": ["linear", "rbf"], "epsilon": [0.1, 0.2]}
    svm_scoring = "r2"

svm_grid = GridSearchCV(svm_model, svm_params, cv=5, scoring=svm_scoring, n_jobs=-1)
svm_grid.fit(X_train_scaled, y_train)
svm_best = svm_grid.best_estimator_
y_pred_svm = svm_best.predict(X_test_scaled)
print("Best parameters:", svm_grid.best_params_)
if TASK == "classification":
    print("Accuracy:", accuracy_score(y_test, y_pred_svm))
    print(classification_report(y_test, y_pred_svm, zero_division=0))
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred_svm)
    plt.show()
else:
    print("MAE:", mean_absolute_error(y_test, y_pred_svm))
    print("RMSE:", mean_squared_error(y_test, y_pred_svm) ** 0.5)
    print("R2:", r2_score(y_test, y_pred_svm))

## 6. K-Nearest Neighbors – KNN
if TASK == "classification":
    knn_model = KNeighborsClassifier()
    knn_params = {"n_neighbors": [3, 5, 7, 9, 11], "weights": ["uniform", "distance"], "p": [1, 2]}
    knn_scoring = "f1_weighted"
else:
    knn_model = KNeighborsRegressor()
    knn_params = {"n_neighbors": [3, 5, 7, 9, 11], "weights": ["uniform", "distance"], "p": [1, 2]}
    knn_scoring = "r2"

knn_grid = GridSearchCV(knn_model, knn_params, cv=5, scoring=knn_scoring, n_jobs=-1)
knn_grid.fit(X_train_scaled, y_train)
knn_best = knn_grid.best_estimator_
y_pred_knn = knn_best.predict(X_test_scaled)
print("Best parameters:", knn_grid.best_params_)
if TASK == "classification":
    print("Accuracy:", accuracy_score(y_test, y_pred_knn))
    print(classification_report(y_test, y_pred_knn, zero_division=0))
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred_knn)
    plt.show()
else:
    print("MAE:", mean_absolute_error(y_test, y_pred_knn))
    print("RMSE:", mean_squared_error(y_test, y_pred_knn) ** 0.5)
    print("R2:", r2_score(y_test, y_pred_knn))

## 7. Multinomial Naive Bayes (Classification only)
# MultinomialNB requires non-negative inputs. X_train_nonnegative uses MinMaxScaler.
mnb_model = MultinomialNB()
mnb_model.fit(X_train_nonnegative, y_train)
y_pred_mnb = mnb_model.predict(X_test_nonnegative)
print("Accuracy:", accuracy_score(y_test, y_pred_mnb))
print(classification_report(y_test, y_pred_mnb, zero_division=0))
ConfusionMatrixDisplay.from_predictions(y_test, y_pred_mnb)
plt.title("Multinomial Naive Bayes – Confusion Matrix")
plt.show()

## 8. Gaussian Naive Bayes (Classification only)
gnb_model = GaussianNB()
gnb_model.fit(X_train_scaled, y_train)
y_pred_gnb = gnb_model.predict(X_test_scaled)
print("Accuracy:", accuracy_score(y_test, y_pred_gnb))
print(classification_report(y_test, y_pred_gnb, zero_division=0))
ConfusionMatrixDisplay.from_predictions(y_test, y_pred_gnb)
plt.title("Gaussian Naive Bayes – Confusion Matrix")
plt.show()

## Model Diagnostics – Classification ROC Curve (Binary classification)
# Run after fitting a classifier that supports predict_proba or decision_function.
# Change `diagnostic_model` to logistic_best, tree_best, knn_best, etc.
diagnostic_model = logistic_best
if TASK == "classification" and len(np.unique(y_test)) == 2:
    if hasattr(diagnostic_model, "predict_proba"):
        diagnostic_scores = diagnostic_model.predict_proba(X_test_scaled)[:, 1] if diagnostic_model is logistic_best else diagnostic_model.predict_proba(X_test_processed)[:, 1]
    elif hasattr(diagnostic_model, "decision_function"):
        diagnostic_scores = diagnostic_model.decision_function(X_test_scaled)
    else:
        diagnostic_scores = None
    if diagnostic_scores is not None:
        fpr, tpr, _ = roc_curve(y_test, diagnostic_scores)
        print("ROC-AUC:", roc_auc_score(y_test, diagnostic_scores))
        plt.plot(fpr, tpr, label="ROC curve")
        plt.plot([0, 1], [0, 1], linestyle="--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.legend()
        plt.show()
else:
    print("ROC curve is shown only for binary classification with probability/score output.")

## Learning Curve (choose the model and matching feature matrix)
# Example below uses Logistic Regression for classification and Linear Regression for regression.
if TASK == "classification":
    learning_model = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE)
    learning_X = X_train_scaled
    learning_scoring = "f1_weighted"
else:
    learning_model = LinearRegression()
    learning_X = X_train_processed
    learning_scoring = "r2"

train_sizes, train_scores, validation_scores = learning_curve(
    learning_model, learning_X, y_train, cv=5, scoring=learning_scoring,
    train_sizes=np.linspace(0.1, 1.0, 5), n_jobs=-1
)
plt.plot(train_sizes, train_scores.mean(axis=1), marker="o", label="Training score")
plt.plot(train_sizes, validation_scores.mean(axis=1), marker="o", label="Cross-validation score")
plt.xlabel("Training examples")
plt.ylabel(learning_scoring)
plt.title("Learning Curve")
plt.legend()
plt.show()

## Feature Importance and SHAP (Optional)
# Built-in tree feature importance (run after Decision Tree training)
if "tree_best" in globals() and hasattr(tree_best, "feature_importances_"):
    tree_importance = pd.Series(tree_best.feature_importances_, index=X_train_processed.columns).sort_values(ascending=False)
    display(tree_importance.head(20).to_frame("Importance"))

# Optional SHAP example for a tree model. Install with: %pip install shap
# import shap
# explainer = shap.TreeExplainer(tree_best)
# shap_values = explainer.shap_values(X_test_processed)
# shap.summary_plot(shap_values, X_test_processed)

# For models without native feature_importances_, use permutation importance:
# from sklearn.inspection import permutation_importance
# perm = permutation_importance(tree_best, X_test_processed, y_test, n_repeats=10, random_state=42)
# display(pd.Series(perm.importances_mean, index=X_test_processed.columns).sort_values(ascending=False).head(20))

# SECTION B – Machine Learning With Pipeline

# Pipeline Data Split and Preprocessor
# Rebuild X and y from the de-duplicated source data.
df_pipe = df.drop_duplicates().dropna(subset=[TARGET]).copy()
X_pipe = df_pipe.drop(columns=[TARGET])
y_pipe = df_pipe[TARGET].copy()

pipe_label_encoder = None
if TASK == "classification" and (y_pipe.dtype == "object" or str(y_pipe.dtype).startswith("category")):
    pipe_label_encoder = LabelEncoder()
    y_pipe = pd.Series(pipe_label_encoder.fit_transform(y_pipe.astype(str)), index=y_pipe.index, name=TARGET)

stratify_pipe = y_pipe if TASK == "classification" and y_pipe.value_counts().min() >= 2 else None
X_train_pipe, X_test_pipe, y_train_pipe, y_test_pipe = train_test_split(
    X_pipe, y_pipe, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=stratify_pipe
)

num_cols_pipe = X_train_pipe.select_dtypes(include=np.number).columns.tolist()
cat_cols_pipe = X_train_pipe.select_dtypes(exclude=np.number).columns.tolist()

numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])
categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_transformer, num_cols_pipe),
    ("cat", categorical_transformer, cat_cols_pipe)
])
print("Numeric columns:", num_cols_pipe)
print("Categorical columns:", cat_cols_pipe)

## 1. Linear Regression Pipeline
linear_pipe = Pipeline(steps=[("preprocess", preprocessor), ("model", LinearRegression())])
linear_pipe.fit(X_train_pipe, y_train_pipe)
y_pred_linear_pipe = linear_pipe.predict(X_test_pipe)
print("MAE:", mean_absolute_error(y_test_pipe, y_pred_linear_pipe))
print("RMSE:", mean_squared_error(y_test_pipe, y_pred_linear_pipe) ** 0.5)
print("R2:", r2_score(y_test_pipe, y_pred_linear_pipe))

## 2. Logistic Regression Pipeline + GridSearchCV
logistic_pipe = Pipeline(steps=[
    ("preprocess", preprocessor),
    ("model", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE))
])
logistic_pipe_grid = GridSearchCV(
    logistic_pipe, {"model__C": [0.01, 0.1, 1, 10]},
    cv=5, scoring="f1_weighted", n_jobs=-1
)
logistic_pipe_grid.fit(X_train_pipe, y_train_pipe)
y_pred_logistic_pipe = logistic_pipe_grid.predict(X_test_pipe)
print("Best parameters:", logistic_pipe_grid.best_params_)
print(classification_report(y_test_pipe, y_pred_logistic_pipe, zero_division=0))
ConfusionMatrixDisplay.from_predictions(y_test_pipe, y_pred_logistic_pipe)
plt.show()

## 3. Decision Tree Pipeline + GridSearchCV
if TASK == "classification":
    tree_pipe_model = DecisionTreeClassifier(random_state=RANDOM_STATE, class_weight="balanced")
    tree_pipe_score = "f1_weighted"
else:
    tree_pipe_model = DecisionTreeRegressor(random_state=RANDOM_STATE)
    tree_pipe_score = "r2"

tree_pipe = Pipeline(steps=[("preprocess", preprocessor), ("model", tree_pipe_model)])
tree_pipe_grid = GridSearchCV(
    tree_pipe,
    {"model__max_depth": [None, 3, 5, 8], "model__min_samples_split": [2, 5, 10]},
    cv=5, scoring=tree_pipe_score, n_jobs=-1
)
tree_pipe_grid.fit(X_train_pipe, y_train_pipe)
y_pred_tree_pipe = tree_pipe_grid.predict(X_test_pipe)
print("Best parameters:", tree_pipe_grid.best_params_)
if TASK == "classification":
    print(classification_report(y_test_pipe, y_pred_tree_pipe, zero_division=0))
    ConfusionMatrixDisplay.from_predictions(y_test_pipe, y_pred_tree_pipe)
    plt.show()
else:
    print("MAE:", mean_absolute_error(y_test_pipe, y_pred_tree_pipe))
    print("RMSE:", mean_squared_error(y_test_pipe, y_pred_tree_pipe) ** 0.5)
    print("R2:", r2_score(y_test_pipe, y_pred_tree_pipe))

## 4. K-Means Pipeline (Unsupervised)
kmeans_pipe = Pipeline(steps=[("preprocess", preprocessor), ("model", KMeans(n_clusters=3, random_state=RANDOM_STATE, n_init=10))])
kmeans_pipe.fit(X_train_pipe)
pipe_train_clusters = kmeans_pipe.predict(X_train_pipe)
pipe_test_clusters = kmeans_pipe.predict(X_test_pipe)
print("Train cluster counts:\n", pd.Series(pipe_train_clusters).value_counts().sort_index())
print("Test cluster counts:\n", pd.Series(pipe_test_clusters).value_counts().sort_index())

### Yellowbrick K-Means Cluster Selection
# Yellowbrick KElbowVisualizer is applied to the transformed training matrix.
# This visualizes candidate cluster counts; fit the final clustering Pipeline separately.
from yellowbrick.cluster import KElbowVisualizer, SilhouetteVisualizer

X_train_pipe_transformed = preprocessor.fit_transform(X_train_pipe)
kmeans_for_viz = KMeans(random_state=RANDOM_STATE, n_init=10)
elbow_visualizer = KElbowVisualizer(kmeans_for_viz, k=(2, 10), metric="distortion")
elbow_visualizer.fit(X_train_pipe_transformed)
elbow_visualizer.show()

# Set BEST_K based on the elbow/silhouette evidence and use it in kmeans_pipe.

## 5. SVM Pipeline + GridSearchCV
if TASK == "classification":
    svm_pipe_model = SVC(class_weight="balanced")
    svm_pipe_params = {"model__C": [0.1, 1, 10], "model__kernel": ["linear", "rbf"]}
    svm_pipe_score = "f1_weighted"
else:
    svm_pipe_model = SVR()
    svm_pipe_params = {"model__C": [0.1, 1, 10], "model__kernel": ["linear", "rbf"]}
    svm_pipe_score = "r2"
svm_pipe = Pipeline(steps=[("preprocess", preprocessor), ("model", svm_pipe_model)])
svm_pipe_grid = GridSearchCV(svm_pipe, svm_pipe_params, cv=5, scoring=svm_pipe_score, n_jobs=-1)
svm_pipe_grid.fit(X_train_pipe, y_train_pipe)
y_pred_svm_pipe = svm_pipe_grid.predict(X_test_pipe)
print("Best parameters:", svm_pipe_grid.best_params_)
if TASK == "classification":
    print(classification_report(y_test_pipe, y_pred_svm_pipe, zero_division=0))
    ConfusionMatrixDisplay.from_predictions(y_test_pipe, y_pred_svm_pipe)
    plt.show()
else:
    print("MAE:", mean_absolute_error(y_test_pipe, y_pred_svm_pipe))
    print("RMSE:", mean_squared_error(y_test_pipe, y_pred_svm_pipe) ** 0.5)
    print("R2:", r2_score(y_test_pipe, y_pred_svm_pipe))

## 6. KNN Pipeline + GridSearchCV
if TASK == "classification":
    knn_pipe_model = KNeighborsClassifier()
    knn_pipe_score = "f1_weighted"
else:
    knn_pipe_model = KNeighborsRegressor()
    knn_pipe_score = "r2"
knn_pipe = Pipeline(steps=[("preprocess", preprocessor), ("model", knn_pipe_model)])
knn_pipe_grid = GridSearchCV(
    knn_pipe,
    {"model__n_neighbors": [3, 5, 7, 9, 11], "model__weights": ["uniform", "distance"]},
    cv=5, scoring=knn_pipe_score, n_jobs=-1
)
knn_pipe_grid.fit(X_train_pipe, y_train_pipe)
y_pred_knn_pipe = knn_pipe_grid.predict(X_test_pipe)
print("Best parameters:", knn_pipe_grid.best_params_)
if TASK == "classification":
    print(classification_report(y_test_pipe, y_pred_knn_pipe, zero_division=0))
    ConfusionMatrixDisplay.from_predictions(y_test_pipe, y_pred_knn_pipe)
    plt.show()
else:
    print("MAE:", mean_absolute_error(y_test_pipe, y_pred_knn_pipe))
    print("RMSE:", mean_squared_error(y_test_pipe, y_pred_knn_pipe) ** 0.5)
    print("R2:", r2_score(y_test_pipe, y_pred_knn_pipe))

## 7. Multinomial Naive Bayes Pipeline
# MultinomialNB must receive non-negative values. Use MinMaxScaler after preprocessing.
mnb_preprocessor = ColumnTransformer(transformers=[
    ("num", Pipeline(steps=[("imputer", SimpleImputer(strategy="median")), ("scale", MinMaxScaler())]), num_cols_pipe),
    ("cat", Pipeline(steps=[("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]), cat_cols_pipe)
])
mnb_pipe = Pipeline(steps=[("preprocess", mnb_preprocessor), ("model", MultinomialNB())])
mnb_pipe_grid = GridSearchCV(mnb_pipe, {"model__alpha": [0.1, 0.5, 1.0, 2.0]}, cv=5, scoring="f1_weighted", n_jobs=-1)
mnb_pipe_grid.fit(X_train_pipe, y_train_pipe)
y_pred_mnb_pipe = mnb_pipe_grid.predict(X_test_pipe)
print("Best parameters:", mnb_pipe_grid.best_params_)
print(classification_report(y_test_pipe, y_pred_mnb_pipe, zero_division=0))
ConfusionMatrixDisplay.from_predictions(y_test_pipe, y_pred_mnb_pipe)
plt.show()

## 8. Gaussian Naive Bayes Pipeline
# GaussianNB expects dense numeric input; OneHotEncoder is configured as dense here.
gnb_preprocessor = ColumnTransformer(transformers=[
    ("num", Pipeline(steps=[("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), num_cols_pipe),
    ("cat", Pipeline(steps=[("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), cat_cols_pipe)
], sparse_threshold=0)
gnb_pipe = Pipeline(steps=[("preprocess", gnb_preprocessor), ("model", GaussianNB())])
gnb_pipe.fit(X_train_pipe, y_train_pipe)
y_pred_gnb_pipe = gnb_pipe.predict(X_test_pipe)
print(classification_report(y_test_pipe, y_pred_gnb_pipe, zero_division=0))
ConfusionMatrixDisplay.from_predictions(y_test_pipe, y_pred_gnb_pipe)
plt.show()

## Pipeline Learning Curve
# Example: classification Logistic Regression pipeline or regression Linear Regression pipeline.
if TASK == "classification":
    curve_estimator = Pipeline(steps=[("preprocess", preprocessor), ("model", LogisticRegression(max_iter=2000))])
    curve_metric = "f1_weighted"
else:
    curve_estimator = Pipeline(steps=[("preprocess", preprocessor), ("model", LinearRegression())])
    curve_metric = "r2"

sizes, train_sc, valid_sc = learning_curve(
    curve_estimator, X_train_pipe, y_train_pipe, cv=5, scoring=curve_metric,
    train_sizes=np.linspace(0.1, 1.0, 5), n_jobs=-1
)
plt.plot(sizes, train_sc.mean(axis=1), marker="o", label="Train")
plt.plot(sizes, valid_sc.mean(axis=1), marker="o", label="Cross-validation")
plt.xlabel("Training examples")
plt.ylabel(curve_metric)
plt.title("Pipeline Learning Curve")
plt.legend()
plt.show()
