import pandas as pd
import seaborn as sns
import plotly.express as px
from sklearn.preprocessing import LabelEncoder

df_transform = pd.read_csv("Data/Transformed Data/training_set_features.csv", index_col=[0])
target_df = pd.read_csv("Data/training_set_labels.csv", index_col=[0])

df_transform.drop(
    ["age_group", "education", "race", "sex", "income_poverty", "marital_status",
     "income_poverty", "employment_status", "hhs_geo_region", "census_msa",
     "employment_industry", "employment_occupation", "rent_or_own"], axis=1, inplace=True)
df_transform = df_transform.join(target_df)
corr = df_transform.corr()
sns.pairplot(df_transform)
fig = px.scatter_matrix(df_transform, dimensions=df_transform.columns)
fig.update_layout(width=800, height=800)
fig.show()

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
df["rent_or_own_label"] = le.fit_transform(df["rent_or_own"])
df["employment_industry_label"] = le.fit_transform(df["employment_industry"])
df["employment_occupation_label"] = le.fit_transform(df["employment_occupation"])
print()
