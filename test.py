import pandas as pd
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv("Data/training_set_features.csv", index_col=[0])
le = LabelEncoder()
df["age_group_label"] = le.fit_transform(df["age_group"])
df["education_label"] = le.fit_transform(df["education"])
df["race_label"] = le.fit_transform(df["race"])
df["sex_label"] = le.fit_transform(df["sex"])
df["income_poverty_label"] = le.fit_transform(df["income_poverty"])
df["marital_status_label"] = le.fit_transform(df["marital_status"])
df["income_poverty_label"] = le.fit_transform(df["rent_or_own"])
df["employment_status_label"] = le.fit_transform(df["employment_status"])
df["hhs_geo_region_label"] = le.fit_transform(df["hhs_geo_region"])
df["census_msa_label"] = le.fit_transform(df["census_msa"])
df["employment_industry_label"] = le.fit_transform(df["employment_industry"])
df["employment_occupation_label"] = le.fit_transform(df["employment_occupation"])
print()
