import pandas as pd
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv('test.csv')


# Check for missing values and count them in each column
missing_values = df.isna().sum()

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
# FireplaceQu 690 - check other rows - it means fireplace quality - previous row is num fireplaces
# GarageType 81  avg
# GarageYrBlt 81  avg
# GarageFinish 81  avg
# GarageQual 81  avg
# GarageCond 81 avg
# PoolQC 1453 - drop column
# Fence 1179  - drop column
# MiscFeature 1406  - drop column


# drop some columns ----------------------------
df.drop(columns=['Alley', 'MasVnrType', 'PoolQC', 'Fence', 'MiscFeature', 'Id'], inplace=True)

# turn data into numbers--------------------
# Initialize the LabelEncoder
label_encoder = LabelEncoder()

# List of columns containing string values that you want to encode
string_columns = df.select_dtypes(include=['object']).columns.tolist()

# Encode the string columns
for col in string_columns:
    df[col] = label_encoder.fit_transform(df[col])

# fix n/a data------------------
# Fill missing values with the mean of each column
for col in df.columns:
    if df[col].dtype != 'object':  # Only for numeric columns
        col_mean = df[col].mean()
        df[col].fillna(col_mean, inplace=True)

for val in df.isna().sum():
    if val != 0:
        print(val)
df.to_csv('cleaned_test.csv', index=False)

