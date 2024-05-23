import pandas as pd
from sklearn.preprocessing import LabelEncoder
import os

data_dir = os.path.join(os.getcwd(), "data")

test_csv = os.path.join(data_dir, "test.csv")
train_csv = os.path.join(data_dir, "train.csv")

test_df = pd.read_csv(test_csv)
train_df = pd.read_csv(train_csv)


# Check for missing values and count them in each column
missing_values = test_df.isna().sum()

# LotFrontage 259 - check data - sometimes doesn't apply, missing for almost half of daya
# Alley 1369  - drop column
# MasVnrType 872 - masonry veneer type - corresponds to a 0 on masvnrarea, drop column
# MasVnrArea 8 - masonry veneer area
# BsmtQual 37 - avg
# BsmtCond 37  avg
# BsmtExposure 38  avg
# BsmtFinType1 37 avg
# BsmtFinType2 38 avg
# Electrical 1 - avg or manually fix
# FireplaceQu 690 - drop
# GarageType 81  avg
# GarageYrBlt 81  avg
# GarageFinish 81  avg
# GarageQual 81  avg
# GarageCond 81 avg
# PoolQC 1453 - drop column
# Fence 1179  - drop column
# MiscFeature 1406  - drop column


# drop some columns ----------------------------
test_df.drop(columns=['Alley', 'MasVnrType', 'PoolQC', 'Fence', 'MiscFeature', 'FireplaceQu', 'Id'], inplace=True)
train_df.drop(columns=['Alley', 'MasVnrType', 'PoolQC', 'Fence', 'MiscFeature', 'FireplaceQu', 'Id'], inplace=True)

# encode data into numbers--------------------
# Initialize the LabelEncoder
label_encoder = LabelEncoder()

# List of columns containing string values that you want to encode
test_string_columns = test_df.select_dtypes(include=['object']).columns.tolist()

# Encode the string columns
for col in test_string_columns:
    test_df[col] = label_encoder.fit_transform(test_df[col])
    
train_string_columns = train_df.select_dtypes(include=['object']).columns.tolist()
for col in train_string_columns:
    train_df[col] = label_encoder.fit_transform(train_df[col])

# fix n/a data------------------
# Fill missing values with the mean of each column
for col in test_df.columns:
    if test_df[col].dtype != 'object':  # Only for numeric columns
        col_mean = test_df[col].mean()
        test_df[col].fillna(col_mean, inplace=True)
        
for col in train_df.columns:
    if train_df[col].dtype != 'object' and col != 'SalePrice':  
        col_mean = train_df[col].mean()
        train_df[col].fillna(col_mean, inplace=True)
        
# scale data -------------------
test_df_scaled = (test_df - test_df.mean()) / test_df.std()

# select all columns to scale except SalePrice
columns_to_scale = train_df.columns.difference(['SalePrice'])

# Apply standardization to selected columns using the std deviation
train_df_scaled = train_df.copy()
train_df_scaled[columns_to_scale] = (train_df[columns_to_scale] - train_df[columns_to_scale].mean()) / train_df[columns_to_scale].std()

# save to csv files --------------
os.chdir(os.path.join(os.getcwd(), "data"))

test_df_scaled.to_csv('cleaned_test.csv', index=False)

# take a chunk of training and make it into validation
val_size = 200

val_df_scaled = pd.DataFrame(train_df_scaled.iloc[:val_size])
train_df_scaled = train_df_scaled.iloc[val_size:]

train_df_scaled.to_csv('cleaned_train.csv', index=False)
val_df_scaled.to_csv('cleaned_val.csv', index=False)

