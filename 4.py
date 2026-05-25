import numpy as np
import pandas as pd
import time
import pickle
import kagglehub
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
import seaborn as sns
import matplotlib.pyplot as plt

tf.random.set_seed(42)
np.random.seed(42)

# -------------------------
# 1. LOAD DATA
# -------------------------
print("Loading data...")
path = kagglehub.dataset_download("kartik2112/fraud-detection")
train_df = pd.read_csv(path + r'\fraudTrain.csv')
test_df = pd.read_csv(path + r'\fraudTest.csv')

print("Train shape:", train_df.shape)
print("Test shape:", test_df.shape)
print("\nTrain fraud rate:", train_df['is_fraud'].mean())
print("Test fraud rate:", test_df['is_fraud'].mean())

reference_date = pd.to_datetime(train_df['trans_date_trans_time']).max()

# -------------------------
# 2. PREPROCESSING
# -------------------------
def preprocess(df, target_encoders=None, scaler=None, fit=False):
    df = df.copy()
    df = df.drop(columns=['Unnamed: 0', 'trans_num', 'first', 'last', 'street'])
    df['trans_date_trans_time'] = pd.to_datetime(df['trans_date_trans_time'])
    df['hour'] = df['trans_date_trans_time'].dt.hour
    df['day_of_week'] = df['trans_date_trans_time'].dt.dayofweek
    df['dob'] = pd.to_datetime(df['dob'])
    df['age'] = (reference_date - df['dob']).dt.days // 365
    df = df.drop(columns=['dob'])
    df['distance'] = np.sqrt((df['lat'] - df['merch_lat'])**2 +
                              (df['long'] - df['merch_long'])**2)
    df = df.drop(columns=['lat', 'long', 'merch_lat', 'merch_long'])
    df = df.sort_values(['cc_num', 'trans_date_trans_time'])
    df['tx_velocity'] = df.groupby('cc_num')['trans_date_trans_time'].transform(
        lambda x: x.expanding().count()
    )
    df['avg_amt'] = df.groupby('cc_num')['amt'].transform('mean')
    df['amt_deviation'] = (df['amt'] - df['avg_amt']).abs() / (df['avg_amt'] + 1)
    df = df.drop(columns=['avg_amt'])
    df['date'] = df['trans_date_trans_time'].dt.date
    df['tx_per_day'] = df.groupby(['cc_num', 'date'])['amt'].transform('count')
    df = df.drop(columns=['date', 'trans_date_trans_time', 'unix_time', 'cc_num', 'city', 'zip'])
    df['gender'] = df['gender'].map({'M': 0, 'F': 1})
    cat_cols = ['job', 'category', 'merchant', 'state']
    if fit:
        target_encoders = {}
        for col in cat_cols:
            fraud_rate = df.groupby(col)['is_fraud'].mean()
            target_encoders[col] = fraud_rate
            df[col] = df[col].map(fraud_rate)
    else:
        for col in cat_cols:
            df[col] = df[col].map(target_encoders[col])
            df[col] = df[col].fillna(target_encoders[col].mean())
    X = df.drop(columns=['is_fraud'])
    y = df['is_fraud']
    if fit:
        scaler = StandardScaler()
        X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    else:
        X = pd.DataFrame(scaler.transform(X), columns=X.columns)
    return X, y, target_encoders, scaler

print("\nPreprocessing...")
start = time.time()
X_train, y_train, target_encoders, scaler = preprocess(train_df, fit=True)
X_test, y_test, _, _ = preprocess(test_df, target_encoders=target_encoders, scaler=scaler, fit=False)
print(f"Time: {(time.time()-start):.1f}s")
print("Features:", X_train.columns.tolist())

# -------------------------
# 3. TRAIN / CV SPLIT (natural split, no fraud moving)
# -------------------------
X_train_split, X_cv, y_train_split, y_cv = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42, stratify=y_train)

X_train_split = X_train_split.reset_index(drop=True)
X_cv = X_cv.reset_index(drop=True)
y_train_split = y_train_split.reset_index(drop=True)
y_cv = y_cv.reset_index(drop=True)

# Autoencoder trains on legit only
X_train_legit = X_train_split[y_train_split == 0].reset_index(drop=True)

print("\nTrain (legit only for autoencoder):", X_train_legit.shape)
print("Train full (for XGBoost):", X_train_split.shape)
print("CV:", X_cv.shape)
print("Fraud in train:", y_train_split.sum())
print("Fraud in CV:", y_cv.sum())

# -------------------------
# 4. AUTOENCODER
# -------------------------

input_dim = X_train_legit.shape[1]
input_layer = Input(shape=(input_dim,))
encoded = Dense(32, activation='relu')(input_layer)
encoded = Dense(16, activation='relu')(encoded)
encoded = Dense(8, activation='relu')(encoded)
decoded = Dense(16, activation='relu')(encoded)
decoded = Dense(32, activation='relu')(decoded)
decoded = Dense(input_dim, activation='linear')(decoded)
autoencoder = Model(input_layer, decoded)
encoder = Model(input_layer, encoded)
autoencoder.compile(optimizer='adam', loss='mse')
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
start = time.time()
history = autoencoder.fit(
    X_train_legit, X_train_legit,
    epochs=50, batch_size=256,
    validation_split=0.1,
    callbacks=[early_stop], verbose=1
)
print(f"Autoencoder training time: {(time.time()-start):.1f}s")
autoencoder.save('autoencoder_fraud.keras')
encoder.save('encoder_fraud.keras')
print("Autoencoder saved!")

# LOAD SAVED AUTOENCODER
autoencoder = tf.keras.models.load_model('autoencoder_fraud.keras')
encoder = tf.keras.models.load_model('encoder_fraud.keras')
print("\nAutoencoder loaded!")

# -------------------------
# 5. GENERATE RECONSTRUCTION ERROR
# -------------------------
print("\nGenerating reconstruction errors...")
start = time.time()

train_pred = autoencoder.predict(X_train_split, verbose=0)
train_mse = np.mean(np.power(X_train_split.values - train_pred, 2), axis=1)

cv_pred = autoencoder.predict(X_cv, verbose=0)
cv_mse = np.mean(np.power(X_cv.values - cv_pred, 2), axis=1)

test_pred = autoencoder.predict(X_test, verbose=0)
test_mse = np.mean(np.power(X_test.values - test_pred, 2), axis=1)

print(f"Time: {(time.time()-start):.1f}s")

# Check separation
train_legit_mse = train_mse[y_train_split == 0]
train_fraud_mse = train_mse[y_train_split == 1]
print(f"\nLegit avg error: {train_legit_mse.mean():.4f}")
print(f"Fraud avg error: {train_fraud_mse.mean():.4f}")

# -------------------------
# 6. BUILD XGBOOST FEATURES
# -------------------------
X_train_xgb = X_train_split.copy()
X_train_xgb['recon_error'] = train_mse
y_train_xgb = y_train_split

X_cv_xgb = X_cv.copy()
X_cv_xgb['recon_error'] = cv_mse

X_test_xgb = X_test.copy()
X_test_xgb['recon_error'] = test_mse

print("\nXGBoost feature shapes:")
print("Train:", X_train_xgb.shape)
print("CV:", X_cv_xgb.shape)
print("Test:", X_test_xgb.shape)

# -------------------------
# 7. SMOTE ON TRAINING DATA
# -------------------------
print("\nApplying SMOTE...")
start = time.time()
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train_xgb, y_train_xgb)
print(f"SMOTE time: {(time.time()-start):.1f}s")
print("Before SMOTE:", y_train_xgb.value_counts().to_dict())
print("After SMOTE:", pd.Series(y_train_smote).value_counts().to_dict())

# -------------------------
# 8. TUNE XGBOOST
# -------------------------
scale = len(y_train_smote[y_train_smote==0]) / len(y_train_smote[y_train_smote==1])

best_f1 = 0
best_params = {}

print("\nTuning XGBoost...")
for n_estimators in [100, 200, 300]:
    for max_depth in [4, 6, 8]:
        for learning_rate in [0.05, 0.1, 0.2]:
            start = time.time()
            xgb = XGBClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=learning_rate,
                scale_pos_weight=scale,
                random_state=42,
                n_jobs=-1
            )
            xgb.fit(X_train_smote, y_train_smote)
            y_prob_cv = xgb.predict_proba(X_cv_xgb)[:, 1]
            best_t = 0
            best_t_f1 = 0
            for t in np.arange(0.1, 0.9, 0.01):
                f1 = f1_score(y_cv, (y_prob_cv > t).astype(int))
                if f1 > best_t_f1:
                    best_t_f1 = f1
                    best_t = t
            print(f"n={n_estimators}, depth={max_depth}, lr={learning_rate} → CV F1: {best_t_f1:.4f} | threshold={best_t:.2f} | Time: {(time.time()-start):.1f}s")
            if best_t_f1 > best_f1:
                best_f1 = best_t_f1
                best_params = {'n_estimators': n_estimators, 'max_depth': max_depth, 'learning_rate': learning_rate, 'threshold': best_t}

print(f"\nBest params: {best_params}")
print(f"Best CV F1: {best_f1:.4f}")

# -------------------------
# 9. TRAIN FINAL MODEL
# -------------------------
print("\nTraining final model...")
xgb_final = XGBClassifier(
    n_estimators=best_params['n_estimators'],
    max_depth=best_params['max_depth'],
    learning_rate=best_params['learning_rate'],
    scale_pos_weight=scale,
    random_state=42,
    n_jobs=-1
)
start = time.time()
xgb_final.fit(X_train_smote, y_train_smote)
print(f"Training time: {(time.time()-start):.1f}s")

best_threshold = best_params['threshold']

# -------------------------
# 10. CV EVALUATION
# -------------------------
y_prob_cv = xgb_final.predict_proba(X_cv_xgb)[:, 1]
y_pred_cv = (y_prob_cv > best_threshold).astype(int)

print("\nCV Results:")
print("F1:", f1_score(y_cv, y_pred_cv))
print("Precision:", precision_score(y_cv, y_pred_cv))
print("Recall:", recall_score(y_cv, y_pred_cv))

# -------------------------
# 11. TEST EVALUATION
# -------------------------
y_prob_test = xgb_final.predict_proba(X_test_xgb)[:, 1]
y_pred_test = (y_prob_test > best_threshold).astype(int)

print("\nTest Results:")
print("F1:", f1_score(y_test, y_pred_test))
print("Precision:", precision_score(y_test, y_pred_test))
print("Recall:", recall_score(y_test, y_pred_test))

# Confusion matrix
cm_test = confusion_matrix(y_test, y_pred_test)
sns.heatmap(cm_test, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Legit', 'Fraud'],
            yticklabels=['Legit', 'Fraud'])
plt.title('Confusion Matrix - Test Set')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show(block=False)
plt.pause(1)

# -------------------------
# 12. SAVE EVERYTHING
# -------------------------
xgb_final.save_model('xgb_model.json')

with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

with open('target_encoders.pkl', 'wb') as f:
    pickle.dump(target_encoders, f)

with open('best_threshold.pkl', 'wb') as f:
    pickle.dump(best_threshold, f)

print("\nAll models saved!")