# Library 
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report, roc_curve, ConfusionMatrixDisplay, f1_score
from sklearn.model_selection import cross_val_score
from xgboost import XGBClassifier
pd.set_option('display.max_rows', None)


def main():

    # DATA UNDERSTANDING

    # Import Data dengan format CSV
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(BASE_DIR, "..", "data", "raw", "loan_data_2007_2014.csv")

    df = pd.read_csv(data_path) 

    # Melihat Beberapa Data Urutan Atas
    df.head()
    
    # Mengetahui Jumlah Baris dan Kolom
    df.shape

    # Mengetahui Nama Kolom, Jumlah baris Non-Null dan Tipe Data
    df.info()

    # Melihat Statistik Deskriptif data Numerikal
    df.describe()

    # Melihat Statistik untuk Data Kategorikal
    df.describe(include="object")

    # Cek Missing Value
    df.isnull().sum().sort_values(ascending=False)

    # Cek Duplicated Value
    df.duplicated().sum()

    # Mengecek Isi Variabel Target & Melihatnya dalam Proporsi
    loan_status_summary = (
        df["loan_status"].value_counts().to_frame()
    )

    loan_status_summary["Percentage (%)"] = (
        df["loan_status"].value_counts(normalize=True) * 100
    ).round(2)

    loan_status_summary



    # EXPLORATORY DATA ANALYSIS (EDA)

    # Kategori yang Digunakan dalam Variabel Target untuk Analisis
    final_status = [
        'Fully Paid',
        'Charged Off',
        'Default',
        'Does not meet the credit policy. Status: Fully Paid',
        'Does not meet the credit policy. Status: Charged Off'
    ]

    df_final = df[df['loan_status'].isin(final_status)].copy()

    # Menentukan Good & Bad dalam credit risk
    df_final['target'] = df_final['loan_status'].map({
        'Fully Paid': 0,
        'Does not meet the credit policy. Status: Fully Paid': 0,
        'Charged Off': 1,
        'Default': 1,
        'Does not meet the credit policy. Status: Charged Off': 1
    })

    # Melihat Proporsi Variabel Target untuk Kategori Good & Bad
    df_final['target'].value_counts()
    df_final['target'].value_counts(normalize=True) * 100


    # Melihat Distribusi dari Variabel Target
    plt.figure(figsize=(6,4))
    sns.countplot(x='target', data=df_final)

    plt.xticks(ticks=[0,1], labels=['Good', 'Bad'])
    plt.title('Distribution of Loan Status')
    plt.xlabel('Loan Status')
    plt.ylabel('Count')

    plt.show()

    # Melihat Variabel yang Memiliki Missing Value Diatas 0%
    missing_percent = (df.isnull().mean() * 100).sort_values(ascending=False)
    missing_percent[missing_percent > 0]

    # Fungsi untuk Analisis Univariate Variabel Numerikal
    def univariate_numerical(df, column, bins=30):

        fig, axes = plt.subplots(1, 2, figsize=(12,5))

        # Histogram
        sns.histplot(df[column], bins=bins, kde=True, ax=axes[0])
        axes[0].set_title(f'Histogram of {column}')
        axes[0].set_xlabel(column)
        axes[0].set_ylabel('Frequency')

        # Boxplot
        sns.boxplot(y=df[column], ax=axes[1])
        axes[1].set_title(f'Boxplot of {column}')
        axes[1].set_xlabel(column)

        plt.tight_layout()
        plt.show()

        # Statistik deskriptif
        print(f"\nDescriptive Statistics for {column}")
        print(df[column].describe())

    # Variabel Numerikal yang Digunakan dalam Analisis
    numerical_cols = [
        'loan_amnt',
        'funded_amnt',
        'int_rate',
        'annual_inc',
        'dti',
        'revol_util',
        'installment',
        'delinq_2yrs',
        'inq_last_6mths',
        'open_acc',
        'pub_rec',
        'revol_bal',
        'total_acc'
    ]

    # Loop untuk Iterasi Semua Variabel Numerikal 
    for col in numerical_cols:
        univariate_numerical(df, col)

    # Variabel Kategorikal yang Digunakan untuk Analisis
    categorical_cols = [
        'home_ownership',
        'verification_status',
        'purpose',
        'grade',
        'sub_grade',
        'term',
        'emp_length'
    ]

    # Fungsi untuk Analisis Univariate Variabel Kategorikal
    def univariate_categorical(df, column):


        data = df[column].value_counts()

        plt.figure(figsize=(12,6))
        sns.barplot(x=data.index, y=data.values)

        plt.title(f'Distribution of {column}')
        plt.xlabel(column)
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    # Loop untuk Iterasi Semua Variabel Kategorikal
    for col in categorical_cols:
        univariate_categorical(df, col)

    # Fungsi untuk Analisis Bivariate Variabel Numerikal vs Target
    def bivariate_numeric_target(df, feature, target):

        plt.figure(figsize=(8,5))
        ax = sns.boxplot(x=target, y=feature, data=df)

        ax.set_xticklabels(['Good', 'Bad'])  

        plt.title(f'{feature} vs Loan Status')
        plt.xlabel('Loan Status')
        plt.ylabel(feature)
        plt.show()

        print("\nMean per Class:")
        print(df.groupby(target)[feature].mean())

    # Loop untuk Iterasi Semua Variabel Numerikal
    for col in numerical_cols:
        bivariate_numeric_target(df_final, col, 'target')


    # Fungsi untuk Analisis Bivariate Variabel Kategorikal vs Target
    def bivariate_categorical_target(df, feature, target='target'):
 
        plt.figure(figsize=(12,5))

        # Mengatur hue_order dan palette supaya 0=Good, 1=Bad
        sns.countplot(
            data=df,
            x=feature,
            hue=target,
            hue_order=[0,1],
            palette={0:'green', 1:'red'}
        )

        plt.title(f'{feature} vs {target}')
        plt.xticks(rotation=45)
        plt.xlabel(feature)
        plt.ylabel('Count')
        plt.legend(title='Loan Status', labels=['Good', 'Bad'])
        plt.show()

        # Proporsi target per kategori
        import pandas as pd
        prop = pd.crosstab(df[feature], df[target], normalize='index') * 100
        print(f"\nProportion of {target} per {feature} (%):")
        print(prop.round(2))

    # Loop untuk Iterasi Semua Variabel Kategorikal
    for col in categorical_cols:
        bivariate_categorical_target(df_final, col, 'target')

    # Correlationn Matrix untuk Variabel Numerikal + Target
    plt.figure(figsize=(12,10))
    sns.heatmap(df_final[numerical_cols + ['target']].corr(), annot=True, fmt=".2f", cmap="coolwarm")
    plt.title("Correlation Matrix")
    plt.show()



    # DATA PREPARATION

    # Drop Variabel Redundan Berdasarkan Correlation Matrix
    numerical_cols.remove('funded_amnt') # Drop Variabel funded_amnt
    categorical_cols.remove('grade') # Drop Variabel grade
    numerical_cols.remove('open_acc') # Drop Variabel open_acc
    numerical_cols.remove('installment') # Drop Variabel Installment

    # Menggabungkan Variabel Numerikal dan Kategorikal
    feature_cols = numerical_cols + categorical_cols

    df_model = df_final[feature_cols + ['target']].copy()

    # Mengecek Info Semua Variabel Setelah di Seleksi
    df_model.info()

    # Mengecek Tipe Data
    df_model.dtypes

    # Menghapus Unsur String dalam Variabel emp_length & Mengubah Tipe Datanya dari String Menjadi Numerik
    df_model['emp_length'] = df_model['emp_length'].str.replace(' years','', regex=False)
    df_model['emp_length'] = df_model['emp_length'].str.replace(' year','', regex=False)
    df_model['emp_length'] = df_model['emp_length'].str.replace('+','', regex=False)
    df_model['emp_length'] = df_model['emp_length'].replace('< 1', '0')
    df_model['emp_length'] = df_model['emp_length'].replace('nan', None)

    df_model['emp_length'] = pd.to_numeric(df_model['emp_length'], errors='coerce')
    df_model['emp_length'].unique()

    # Menghapus Unsur String dalam Variabel term & Mengubah Tipe Datanya dari String Menjadi Numerik
    df_model['term'] = (
        df_model['term'].astype(str).str.extract('(\d+)')
    )

    df_model['term'] = pd.to_numeric(df_model['term'], errors='coerce')
    df_model['term'].unique()

    # Mengecek Tipe Data Setelah Melakukan Cleaning
    df_model.dtypes

    # Split Data dengan Ukuran 80:20
    X = df_model.drop('target', axis=1)
    y = df_model['target']


    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # Memisahkan Numerical dan Categorical 
    numerical_cols = X_train.select_dtypes(include=['int64','float64']).columns
    categorical_cols = X_train.select_dtypes(include=['object']).columns

    # Imputasi untuk Variabel Numerikal dengan Mengganti Missing Value Menjadi Median dari Variabel Tersebut
    for col in numerical_cols:
        median_value = X_train[col].median()
        X_train[col] = X_train[col].fillna(median_value)
        X_test[col] = X_test[col].fillna(median_value)

    # Imputasi untuk Variabel Kategorikal dengan Mengganti Missing Value Menjadi Unknown 
    for col in categorical_cols:
        X_train[col] = X_train[col].fillna('Unknown')
        X_test[col] = X_test[col].fillna('Unknown')

    # Memastikan Tidak Ada Missing Value Lagi
    X_train.isna().sum().sum()
    X_test.isna().sum().sum()

    # CAPPING (WINSORIZING)
    for col in numerical_cols:
        lower = X_train[col].quantile(0.01)
        upper = X_train[col].quantile(0.99)

        X_train[col] = X_train[col].clip(lower, upper)
        X_test[col] = X_test[col].clip(lower, upper)

    # Merubah Variabel Kategorikal menjadi Biner dengan Metode One-Hot Encoding
    X_train = pd.get_dummies(X_train, columns=categorical_cols, drop_first=True)
    X_test = pd.get_dummies(X_test, columns=categorical_cols, drop_first=True)

    # Samakan kolom train & test
    X_train, X_test = X_train.align(X_test, join='left', axis=1, fill_value=0)

    # Membuat Fitur Baru berdasarkan Fitur yang Telah Ada (Feature Engineering)
    for df in [X_train, X_test]:
        df['loan_to_income'] = df['loan_amnt'] / df['annual_inc'] # Loan to Income Ratio
        df['log_annual_inc'] = np.log1p(df['annual_inc']) # Log Transformation for Annual Income
        df['dti_x_revol'] = df['dti'] * df['revol_util'] # Interaction dti and revol_util
        df['emp_stability_ratio'] = df['emp_length'] / df['term'] # Employment stability
        df['credit_behavior_score'] = df['delinq_2yrs'] + df['inq_last_6mths'] + df['pub_rec'] # Behavior Aggregation

    # Standarisasi Data
    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)



    # DATA MODELING
    
    # LOGISTIC REGRESSION
    # Training
    logreg = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
    logreg.fit(X_train_scaled, y_train)

    # Prediction
    y_pred_log = logreg.predict(X_test_scaled)
    y_proba_log = logreg.predict_proba(X_test_scaled)[:,1]

    # Evaluation Metrics for LR
    print("ROC-AUC:", roc_auc_score(y_test, y_proba_log))
    print("\nLogistic Regression Classification Report:\n", classification_report(y_test, y_pred_log))

    # Overfitting Check
    train_auc_log = roc_auc_score(
        y_train,
        logreg.predict_proba(X_train_scaled)[:,1]
    )

    test_auc_log = roc_auc_score(
        y_test,
        logreg.predict_proba(X_test_scaled)[:,1]
    )

    print("Train AUC Logistic:", train_auc_log)
    print("Test AUC Logistic:", test_auc_log)

    # Confusion Matrix
    plt.figure(figsize=(4,4))
    ConfusionMatrixDisplay.from_predictions(
        y_test,
        y_pred_log,
        cmap="Blues"
    )
    plt.title("Confusion Matrix - Logistic Regression")
    plt.show()

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_proba_log)

    plt.figure(figsize=(6,5))
    plt.plot(fpr, tpr, label="Logistic")
    plt.plot([0,1],[0,1],'k--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve - Logistic Regression")
    plt.legend()
    plt.show()


    # RANDOM FOREST
    # Training
    rf = RandomForestClassifier(
        n_estimators=400,
        max_depth=10,
        min_samples_split=10,
        min_samples_leaf=5,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )

    rf.fit(X_train, y_train)

    # Prediction
    y_pred_rf = rf.predict(X_test)
    y_proba_rf = rf.predict_proba(X_test)[:,1]

    # Evaluation Metrics for RF
    print("ROC-AUC RF:", roc_auc_score(y_test, y_proba_rf))
    print("\nRandom Forest Classification Report:\n", classification_report(y_test, y_pred_rf))

    # Overfitting Check
    train_auc_rf = roc_auc_score(y_train, rf.predict_proba(X_train)[:,1])
    test_auc_rf = roc_auc_score(y_test, y_proba_rf)

    print("Train AUC:", train_auc_rf)
    print("Test AUC:", test_auc_rf)

    # Confusion Matrix
    plt.figure(figsize=(4,4))
    ConfusionMatrixDisplay.from_predictions(
        y_test,
        y_pred_rf,
        cmap="Greens"
    )
    plt.title("Confusion Matrix - Random Forest")
    plt.show()

    # ROC Curve
    fpr_rf, tpr_rf, _ = roc_curve(y_test, y_proba_rf)

    # Plot
    plt.figure(figsize=(6,5))
    plt.plot(fpr_rf, tpr_rf, label="Random Forest")
    plt.plot([0,1],[0,1],'k--')  # garis random

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve - Random Forest")
    plt.legend()
    plt.show()


    # XGBOOST
    # Training
    scale_pos_weight = len(y_train[y_train==0]) / len(y_train[y_train==1])

    xgb = XGBClassifier(
        n_estimators=500,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        eval_metric='logloss'
    )

    xgb.fit(X_train, y_train)

    # Prediction
    y_pred_xgb = xgb.predict(X_test)
    y_proba_xgb = xgb.predict_proba(X_test)[:,1]

    # Evalution Metrics for XGBoost
    print("ROC-AUC XGB:", roc_auc_score(y_test, y_proba_xgb))
    print("\nClassification Report:\n", classification_report(y_test, y_pred_xgb))

    # Overfitting Check
    train_auc_xgb = roc_auc_score(
        y_train, xgb.predict_proba(X_train)[:,1]
    )

    test_auc_xgb = roc_auc_score(
        y_test, y_proba_xgb
    )

    print("Train AUC XGB:", train_auc_xgb)
    print("Test AUC XGB:", test_auc_xgb)

    # Confusion Matrix
    plt.figure(figsize=(4,4))
    ConfusionMatrixDisplay.from_predictions(
        y_test,
        y_pred_xgb,
        cmap="Oranges"
    )
    plt.title("Confusion Matrix - XGBoost")
    plt.show()

    # ROC Curve
    fpr_xgb, tpr_xgb, _ = roc_curve(y_test, y_proba_xgb)

    plt.figure(figsize=(6,5))
    plt.plot(fpr_xgb, tpr_xgb, label="XGBoost")
    plt.plot([0,1],[0,1],'k--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve - XGBoost")
    plt.legend()
    plt.show()



    # MODEL EVALUATION

    # PERBANDINGAN AUC, RECALL, PRECISION, F1-SCORE
    # Logistic
    report_log = classification_report(y_test, y_pred_log, output_dict=True)

    # RF
    report_rf = classification_report(y_test, y_pred_rf, output_dict=True)

    # XGB       
    report_xgb = classification_report(y_test, y_pred_xgb, output_dict=True)

    # Tabel Perbandingan Metrik Evaluasi Ketiga Algoritma
    comparison_df = pd.DataFrame({
        "Model": ["Logistic", "Random Forest", "XGBoost"],

        "Train AUC": [
            train_auc_log,
            train_auc_rf,
            train_auc_xgb
        ],

        "Test AUC": [
            test_auc_log,
            test_auc_rf,
            test_auc_xgb
        ],

        "AUC Gap": [
            train_auc_log - test_auc_log,
            train_auc_rf - test_auc_rf,
            train_auc_xgb - test_auc_xgb
        ],

        "Recall (Bad)": [
            report_log["1"]["recall"],
            report_rf["1"]["recall"],
            report_xgb["1"]["recall"]
        ],

        "Precision (Bad)": [
            report_log["1"]["precision"],
            report_rf["1"]["precision"],
            report_xgb["1"]["precision"]
        ],

        "F1-Score (Bad)": [
            report_log["1"]["f1-score"],
            report_rf["1"]["f1-score"],
            report_xgb["1"]["f1-score"]
        ]

    })

    comparison_df

    # Perbandingan Confusion Matrix Ketiga Algoritma
    fig, axes = plt.subplots(1, 3, figsize=(15,4))

    models = [
        ("Logistic", y_pred_log),
        ("Random Forest", y_pred_rf),
        ("XGBoost", y_pred_xgb)
    ]

    for ax, (name, y_pred) in zip(axes, models):
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", ax=ax)
        ax.set_title(name)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")

    plt.tight_layout()
    plt.show()

    # Perbandingan ROC Curve Ketiga Algoritma
    fpr_log, tpr_log, _ = roc_curve(y_test, y_proba_log)
    fpr_rf, tpr_rf, _ = roc_curve(y_test, y_proba_rf)
    fpr_xgb, tpr_xgb, _ = roc_curve(y_test, y_proba_xgb)

    plt.figure(figsize=(8,6))
    plt.plot(fpr_log, tpr_log, label="Logistic")
    plt.plot(fpr_rf, tpr_rf, label="Random Forest")
    plt.plot(fpr_xgb, tpr_xgb, label="XGBoost")
    plt.plot([0,1],[0,1],'k--')

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")    
    plt.title("ROC Curve Comparison")
    plt.legend()
    plt.show()


    # Berdasarkan modeling ditentukan bahwa XGBoost dipilih sebagai model terbaik

# Entry point
if __name__ == "__main__":
    main()