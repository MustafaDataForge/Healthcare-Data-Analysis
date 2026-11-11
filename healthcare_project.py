import numpy as np
import pandas as pd
df=pd.read_csv("healthcare_dataset.csv")
print(df.head())
print(df.columns)
print(df.info())
print(df.shape)
print(df.isnull().sum())
print(df["Medical Condition"].unique())
patients= df[["Name","Age","Medical Condition","Billing Amount"]]
print(patients.columns)
print(df[(df["Medical Condition"]=="Diabetes")&(df["Age"]>50)])
df=df.drop_duplicates()
print(df.shape)
df=df[df["Billing Amount"]>=0]
print(df.shape)
df["Name"]=df["Name"].str.title()
print(df["Name"])
df["Date of Admission"]=pd.to_datetime(df["Date of Admission"])
df["Discharge Date"]=pd.to_datetime(df["Discharge Date"])
print(df["Date of Admission"].dtype)
df["Length of Stay"]=df["Discharge Date"]-df["Date of Admission"]
print(df.head())

def get_age_group(age):
    if age <= 12:
     return "Child"
    elif age <=19:
       return "Teen"
    elif age <=59:
       return "Adult"
    else:
       return "Senior"

df["Age Group"]=df["Age"].apply(get_age_group)
print(df.head())

avg_billing=df["Billing Amount"].mean()
df["High Bill"]=df["Billing Amount"].apply(lambda x :"Yes" if x > avg_billing else "No")
top10=df.sort_values(by="Billing Amount",ascending=False).head(10)
print(top10[["Name","Medical Condition","Billing Amount"]])

conditional_summary=df.groupby("Medical Condition").agg(
   average_billing=("Billing Amount","mean"),
   average_stay=("Length of Stay","mean")
)
print(conditional_summary)


import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")

# 1. Patients by Medical Condition
plt.figure(figsize=(10, 5))
sns.countplot(data=df, x="Medical Condition")
plt.title("Patients by Medical Condition")
plt.xlabel("Medical Condition")
plt.ylabel("Number of Patients")
plt.xticks(rotation=45)
plt.show()


# 2. Patient Age Distribution
plt.figure(figsize=(8, 5))
sns.histplot(data=df, x="Age", bins=20, kde=True)
plt.title("Patient Age Distribution")
plt.xlabel("Age")
plt.ylabel("Number of Patients")
plt.show()


# 3. Average Billing by Medical Condition
plt.figure(figsize=(10, 5))
sns.barplot(data=df, x="Medical Condition",
            y="Billing Amount", errorbar=None)
plt.title("Average Billing by Medical Condition")
plt.xlabel("Medical Condition")
plt.ylabel("Average Billing Amount")
plt.xticks(rotation=45)
plt.show()


# 4. Patients by Age Group
plt.figure(figsize=(8, 5))
sns.countplot(data=df, x="Age Group",
              order=["Child", "Teen", "Adult", "Senior"])
plt.title("Patients by Age Group")
plt.xlabel("Age Group")
plt.ylabel("Number of Patients")
plt.show()


# 5. Billing Amount Outlier Detection
plt.figure(figsize=(9, 4))
sns.boxplot(data=df, x="Billing Amount")
plt.title("Billing Amount Outlier Detection")
plt.xlabel("Billing Amount")
plt.show()


# 6. Average Hospital Stay by Medical Condition
plt.figure(figsize=(10, 5))
sns.barplot(data=df, x="Medical Condition",
            y="Length of Stay", errorbar=None)
plt.title("Average Hospital Stay by Medical Condition")
plt.xlabel("Medical Condition")
plt.ylabel("Average Stay (Days)")
plt.xticks(rotation=45)
plt.show()


# 7. Correlation Heatmap
plt.figure(figsize=(9, 6))
sns.heatmap(
    df[["Age", "Billing Amount", "Length of Stay"]].corr(),
    annot=True,
    cmap="coolwarm"
)
plt.title("Healthcare Data Correlation")
plt.show()
