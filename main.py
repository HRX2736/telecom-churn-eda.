# ==============================
# TELECOM CHURN - FULL EDA PROJECT (10 Points)
# ==============================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ------------------------------
# 1) Load Dataset + Check Loaded
# ------------------------------
file_path = "telecom_churn.csv"

try:
    df = pd.read_csv(file_path)
    print("✅ File loaded successfully!")
    print("Shape:", df.shape)
except Exception as e:
    print("❌ Error loading file:", e)

print("\n✅ First 5 rows:")
print(df.head())

# ------------------------------
# 2) Basic Info & Structure
# ------------------------------
print("\n✅ Data Info:")
print(df.info())

print("\n✅ Columns:")
print(df.columns)

print("\n✅ Data Types:")
print(df.dtypes)

# ------------------------------
# 3) Check Missing Values
# ------------------------------
print("\n✅ Missing Values (Count):")
print(df.isnull().sum())

print("\n✅ Missing Values (%):")
print((df.isnull().sum() / len(df)) * 100)

# ------------------------------
# 4) Check Duplicates
# ------------------------------
print("\n✅ Duplicate rows:", df.duplicated().sum())

# Remove duplicates if any
df = df.drop_duplicates()
print("✅ After removing duplicates, Shape:", df.shape)

# ------------------------------
# 5) Fix Column Issues + Convert Types
# ------------------------------
# TotalCharges in this dataset often has blanks stored as " "
if "TotalCharges" in df.columns:
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

# Fill missing values in TotalCharges (if any)
if df["TotalCharges"].isnull().sum() > 0:
    df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())
    print("✅ Missing TotalCharges fixed using median.")

# ------------------------------
# 6) Clean / Standardize Values
# ------------------------------
# Convert SeniorCitizen to category label for better readability
if "SeniorCitizen" in df.columns:
    df["SeniorCitizen"] = df["SeniorCitizen"].replace({0: "No", 1: "Yes"})

# Strip spaces in object columns
for col in df.select_dtypes(include="object").columns:
    df[col] = df[col].astype(str).str.strip()

print("\n✅ Data cleaned and standardized!")

# ------------------------------
# 7) Quick EDA: Churn Distribution
# ------------------------------
print("\n✅ Churn Value Counts:")
print(df["Churn"].value_counts())

print("\n✅ Churn Percentage:")
print(df["Churn"].value_counts(normalize=True) * 100)

plt.figure(figsize=(6, 4))
df["Churn"].value_counts().plot(kind="bar")
plt.title("Churn Count")
plt.xlabel("Churn")
plt.ylabel("Customers")
plt.show()

# ------------------------------
# 8) Key Insights (10 Best Insights)
# ------------------------------
print("\n==============================")
print("✅ TOP 10 QUICK INSIGHTS")
print("==============================")

# Insight 1: Contract vs Churn
contract_churn = pd.crosstab(df["Contract"], df["Churn"], normalize="index") * 100
print("\n1) Contract vs Churn %:\n", contract_churn)

contract_churn.plot(kind="bar", figsize=(8, 4))
plt.title("Churn % by Contract Type")
plt.ylabel("Percentage")
plt.show()

# Insight 2: Tenure vs Churn
plt.figure(figsize=(8, 4))
df[df["Churn"] == "No"]["tenure"].plot(kind="hist", bins=30, alpha=0.7, label="No")
df[df["Churn"] == "Yes"]["tenure"].plot(kind="hist", bins=30, alpha=0.7, label="Yes")
plt.title("Tenure Distribution by Churn")
plt.xlabel("Tenure (Months)")
plt.legend()
plt.show()

# Insight 3: MonthlyCharges vs Churn
plt.figure(figsize=(8, 4))
df[df["Churn"] == "No"]["MonthlyCharges"].plot(kind="hist", bins=30, alpha=0.7, label="No")
df[df["Churn"] == "Yes"]["MonthlyCharges"].plot(kind="hist", bins=30, alpha=0.7, label="Yes")
plt.title("Monthly Charges Distribution by Churn")
plt.xlabel("MonthlyCharges")
plt.legend()
plt.show()

# Insight 4: InternetService vs Churn
internet_churn = pd.crosstab(df["InternetService"], df["Churn"], normalize="index") * 100
print("\n4) Internet Service vs Churn %:\n", internet_churn)

internet_churn.plot(kind="bar", figsize=(8, 4))
plt.title("Churn % by Internet Service")
plt.ylabel("Percentage")
plt.show()

# Insight 5: PaymentMethod vs Churn
payment_churn = pd.crosstab(df["PaymentMethod"], df["Churn"], normalize="index") * 100
print("\n5) Payment Method vs Churn %:\n", payment_churn)

payment_churn.plot(kind="bar", figsize=(10, 4))
plt.title("Churn % by Payment Method")
plt.ylabel("Percentage")
plt.show()

# Insight 6: PaperlessBilling vs Churn
paperless_churn = pd.crosstab(df["PaperlessBilling"], df["Churn"], normalize="index") * 100
print("\n6) Paperless Billing vs Churn %:\n", paperless_churn)

paperless_churn.plot(kind="bar", figsize=(6, 4))
plt.title("Churn % by Paperless Billing")
plt.ylabel("Percentage")
plt.show()

# Insight 7: TechSupport vs Churn
techsupport_churn = pd.crosstab(df["TechSupport"], df["Churn"], normalize="index") * 100
print("\n7) Tech Support vs Churn %:\n", techsupport_churn)

techsupport_churn.plot(kind="bar", figsize=(8, 4))
plt.title("Churn % by Tech Support")
plt.ylabel("Percentage")
plt.show()

# Insight 8: OnlineSecurity vs Churn
security_churn = pd.crosstab(df["OnlineSecurity"], df["Churn"], normalize="index") * 100
print("\n8) Online Security vs Churn %:\n", security_churn)

security_churn.plot(kind="bar", figsize=(8, 4))
plt.title("Churn % by Online Security")
plt.ylabel("Percentage")
plt.show()

# Insight 9: SeniorCitizen vs Churn
senior_churn = pd.crosstab(df["SeniorCitizen"], df["Churn"], normalize="index") * 100
print("\n9) Senior Citizen vs Churn %:\n", senior_churn)

senior_churn.plot(kind="bar", figsize=(6, 4))
plt.title("Churn % by Senior Citizen")
plt.ylabel("Percentage")
plt.show()

# Insight 10: Average Charges by Churn
avg_charges = df.groupby("Churn")[["MonthlyCharges", "TotalCharges", "tenure"]].mean()
print("\n10) Avg MonthlyCharges, TotalCharges, Tenure by Churn:\n", avg_charges)

# ------------------------------
# 9) Create Simple Risk Flag
# ------------------------------
df["HighRisk"] = np.where(
    (df["Contract"] == "Month-to-month") &
    (df["tenure"] < 12) &
    (df["MonthlyCharges"] > df["MonthlyCharges"].median()),
    "Yes",
    "No"
)

print("\n✅ HighRisk flag created!")
print(df["HighRisk"].value_counts())

# ------------------------------
# 10) Final Output + Export Clean File
# ------------------------------
print("\n✅ Final Cleaned Dataset Shape:", df.shape)

output_file = "telecom_churn_cleaned.csv"
df.to_csv(output_file, index=False)
print(f"✅ Cleaned file saved as: {output_file}")

print("\n✅ Project Completed Successfully 🔥")

