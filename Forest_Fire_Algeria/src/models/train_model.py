import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression,SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score,confusion_matrix,recall_score,precision_score
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

##load dataset
df = pd.read_pickle('../../data/external/build_features_pca.pkl')

## create training and testing set
df_train = df.drop(['Region', 'Date'],axis=1)
X = df_train.drop(['Classes'],axis=1)
Y = df_train.drop(X,axis=1)

##train test split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.25,stratify=Y)


fig, ax = plt.subplots(figsize=(10, 5))
df_train["Classes"].value_counts().plot(
    kind="bar", ax=ax, color="lightblue", label="Total"
)
Y_train.value_counts().plot(kind="bar", ax=ax, color="dodgerblue", label="Train")
Y_test.value_counts().plot(kind="bar", ax=ax, color="royalblue", label="Test")
plt.legend()
plt.show()


print(f"Total: {len(X)}")  
print(f"Train: {len(X_train)} ({len(X_train)/len(X):.1%})")  
print(f"Test: {len(X_test)} ({len(X_test)/len(X):.1%})")  


### split features subset

Basic_features = ['Temperature', 'RH', 'Ws', 'Rain','FFMC', 'DMC', 'DC', 'ISI', 'BUI','FWI']
PCA_features =['pca_1', 'pca_2', 'pca_3', 'pca_4'] 


feature_set_1 = list(Basic_features)
feature_set_2 = list(PCA_features)
feature_set_3 = list(Basic_features + PCA_features)

possible_feature_sets = [feature_set_1, feature_set_2, feature_set_3]
feature_names = ["Basic", "PCA", "Combined"]

models = {
    "Random Forest": RandomForestClassifier,
    "AdaBoost": AdaBoostClassifier,
    "Naive Bayes": GaussianNB,
    "Logistic Regression": LogisticRegression,
    "SGD Classifier": SGDClassifier
}


# Feature sets
possible_feature_sets = [feature_set_1, feature_set_2, feature_set_3]
feature_names = ["Basic", "PCA", "Combined"]

saved_models = {}
score_df = pd.DataFrame()
cv_folds = 5


for i, feature_name in enumerate(feature_names):
    feature_set = possible_feature_sets[i]
    X_train_selected = X_train[feature_set]

    print(f"Feature Set: {feature_name} ({len(feature_set)} features)")

    for model_name, model_class in models.items():
        print(f"\tTraining: {model_name}")

        # Instantiate new model each time
        if model_name == "Logistic Regression":
            model = model_class(max_iter=10000)
        else:
            model = model_class()

        # Build pipeline
        if model_name == "Naive Bayes":
            pipe = Pipeline([("classifier", model)])
        else:
            pipe = Pipeline([
                ("scaler", StandardScaler()),
                ("classifier", model)
            ])

        # Train
        pipe.fit(X_train_selected, Y_train)

        # Save model and metrics
        model_key = f"{model_name}_{feature_name}"
        saved_models[model_key] = {
            'pipe': pipe,
            'features': list(X_train_selected.columns),
            'model_name': model_name,
            'feature_set_name': feature_name
        }

        # Evaluate & log score
        train_acc = accuracy_score(Y_train, pipe.predict(X_train_selected))
        cv_scores = cross_val_score(pipe, X_train_selected, Y_train, cv=5, scoring='accuracy')

        new_score = pd.DataFrame([{
            "model": model_name,
            "set_of_features": feature_name,
            "num_features": len(feature_set),
            "train_accuracy": train_acc,
            "cv_mean_accuracy": cv_scores.mean(),
            "cv_std_accuracy": cv_scores.std()
        }])
        score_df = pd.concat([score_df, new_score], ignore_index=True)

print("\nTraining complete.")

#checking results

score_df['Gap'] = score_df['train_accuracy'] - score_df['cv_mean_accuracy']
score_df.sort_values(by=['train_accuracy','cv_mean_accuracy'] ,ascending=False)
score_df[score_df['Gap']<0.05].sort_values(by=['train_accuracy','cv_mean_accuracy'],ascending=False)




plt.figure(figsize=(10,10))
sns.barplot(x='model',y='cv_mean_accuracy',hue='set_of_features',data=score_df)
plt.xlabel('Model')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.show()

"""
-from training different models pca feautres and combined features doesnt show much 
 of improvement so Basic features are good.
-models like random forest and adaboost did a good training but need to test them on new data
-models like logisctic regression and SGD are acceptable
       
"""
# Save the entire dictionary to a file

joblib.dump(saved_models, '../../models/saved_models.pkl')


# Get the  best models
rf = saved_models['Random Forest_Basic']['pipe']
rf_feat = saved_models['Random Forest_Basic']['features']

ada = saved_models['AdaBoost_Basic']['pipe']
ada_feat = saved_models['AdaBoost_Basic']['features']
### accuracy score on test data

#Random Forest metrics
Y_pred_RF = rf.predict(X_test[rf_feat])
RF_acc  = accuracy_score(Y_test,Y_pred_RF)
RF_Prec = precision_score(Y_test,Y_pred_RF,pos_label='fire')
RF_recall= recall_score(Y_test,Y_pred_RF,pos_label='fire')
 

#AdaBoost metrics
Y_pred_ada= ada.predict(X_test[ada_feat])
Ada_acc = accuracy_score(Y_test,Y_pred_ada)
Ada_Prec = precision_score(Y_test,Y_pred_ada,pos_label='fire')
Ada_recall = recall_score(Y_test,Y_pred_ada,pos_label='fire')  

    
metrics = ['Accuracy', 'Precision', 'Recall']

RF_scores = [RF_acc, RF_Prec, RF_recall]
Ada_scores = [Ada_acc, Ada_Prec, Ada_recall]

x = np.arange(len(metrics))  
width = 0.35  

fig, ax = plt.subplots(figsize=(8, 5))
rects1 = ax.bar(x - width/2, RF_scores, width, label='Random Forest', color='skyblue')
rects2 = ax.bar(x + width/2, Ada_scores, width, label='AdaBoost', color='salmon')

ax.set_ylabel('Score')
ax.set_title('Model Performance Comparison')
ax.set_xticks(x)
ax.set_xticklabels(metrics)
ax.set_ylim(0, 1.1)  
ax.legend()

for rect in rects1 + rects2:
    height = rect.get_height()
    ax.annotate(f'{height:.2f}',
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha='center', va='bottom')
plt.tight_layout()
plt.savefig('../../reports/figures/Training_model/Model_Perfomance_comparison.png')
plt.show()
plt.close()

##confusion matrix
rf_cm =  confusion_matrix(Y_test,Y_pred_RF)   
ada_cm = confusion_matrix(Y_test,Y_pred_ada)

# Set class labels
class_labels = ['Not Fire', 'Fire']

# Plotting
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Random Forest Confusion Matrix
sns.heatmap(rf_cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=axes[0])
axes[0].set_title('Random Forest\nConfusion Matrix', fontsize=14)
axes[0].set_xlabel('Predicted Label', fontsize=12)
axes[0].set_ylabel('True Label', fontsize=12)
axes[0].set_xticklabels(class_labels)
axes[0].set_yticklabels(class_labels)

# AdaBoost Confusion Matrix
sns.heatmap(ada_cm, annot=True, fmt='d', cmap='Oranges', cbar=False, ax=axes[1])
axes[1].set_title('AdaBoost\nConfusion Matrix', fontsize=14)
axes[1].set_xlabel('Predicted Label', fontsize=12)
axes[1].set_ylabel('True Label', fontsize=12)
axes[1].set_xticklabels(class_labels)
axes[1].set_yticklabels(class_labels)

plt.tight_layout()
plt.savefig('../../reports/figures/Training_model/confusion_matrix.png')
plt.show()
plt.close()
"""
both models have the same performance on the test set 
"""

## save the best model 

joblib.dump(rf, f'../../models/random_forest_model.pkl')
joblib.dump(ada, f'../../models/adaboost_model.pkl')

