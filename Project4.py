# ============================================================
#  Productivity Analysis using Lifestyle Factors (FULL EDA)
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# ── Style Setup ──────────────────────────────────────────────
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

# ============================================================
#  LOAD DATA
# ============================================================
df = pd.read_excel("ScreenTime vs MentalWellness12121.xlsx")

print("=" * 60)
print("   PRODUCTIVITY ANALYSIS - FULL DATASET")
print("=" * 60)
print(f"Dataset Shape: {df.shape}")
print(f"Missing Values: {df.isnull().sum().sum()}")

# ============================================================
#  FEATURE ENGINEERING
# ============================================================

# Productivity level
def productivity_level(x):
    if x >= 70:
        return "High"
    elif x >= 40:
        return "Medium"
    else:
        return "Low"

df['productivity_level'] = df['productivity_0_100'].apply(productivity_level)
order = ["Low", "Medium", "High"]

# Age groups
def age_group(age):
    if age <= 24:
        return "18-24"
    elif age <= 30:
        return "25-30"
    else:
        return "Other"

df['age_group'] = df['age'].apply(age_group)

# ============================================================
#  1. OVERVIEW (DISTRIBUTION)
# ============================================================

plt.figure()
plt.hist(df['productivity_0_100'], bins=25, color='#6baed6', alpha=0.7)
plt.title("Productivity Score Distribution")
plt.xlabel("Score")
plt.ylabel("Frequency")
plt.show()

# 
# ============================================================
#  2. AGE GROUP ANALYSIS
# ============================================================

plt.figure()
sns.barplot(data=df,
            x='age_group',
            y='productivity_0_100',
            color='#6baed6')

plt.title("Productivity by Age Group")
plt.xlabel("Age Group")
plt.ylabel("Avg Productivity")
plt.show()

# ============================================================
#  3. GENDER ANALYSIS
# ============================================================

plt.figure()
sns.boxplot(data=df,
            x='gender',
            y='productivity_0_100',
             color='#bcbddc')   # soft purple

plt.title("Productivity by Gender")
plt.show()

# ============================================================
#  4. OCCUPATION ANALYSIS
# ============================================================

plt.figure(figsize=(12,5))

occ = df.groupby('occupation')['productivity_0_100'].mean().sort_values()

plt.barh(occ.index, occ.values, color='#6baed6')

plt.title("Average Productivity by Occupation")
plt.xlabel("Productivity Score")
plt.show()

# ============================================================
#  5. WORK MODE ANALYSIS
# ============================================================

plt.figure()

sns.boxplot(data=df,
            x='work_mode',
            y='productivity_0_100',
            color='#9ecae1')

plt.title("Productivity by Work Mode")
plt.show()

# ============================================================
#  6. SCATTER RELATIONSHIPS
# ============================================================

plt.figure()

sc = plt.scatter(df['sleep_hours'],
                 df['productivity_0_100'],
                 c=df['stress_level_0_10'],
                 cmap='Blues',
                 alpha=0.6)

plt.colorbar(sc, label="Stress Level")

plt.title("Sleep vs Productivity")
plt.xlabel("Sleep Hours")
plt.ylabel("Productivity")
plt.show()

# # ============================================================
# #  7. BOX PLOTS (KEY FACTORS)
# # ============================================================
plt.figure()

sns.boxplot(data=df,
            x='productivity_level',
            y='sleep_hours',
            order=order,
             color='#a1d99b')   # soft green

plt.title("Sleep vs Productivity Level")
plt.show()


  



# ============================================================
#  8. HEATMAP (ALL NUMERIC FEATURES)
# ============================================================

plt.figure(figsize=(10, 7))
num_cols = df.select_dtypes(include=np.number)

sns.heatmap(num_cols.corr(),
            cmap='Blues',
            annot=True,
            fmt=".2f")

plt.title("Correlation Between All Features")
plt.show()


# ============================================================
#  SUMMARY
# ============================================================

print("\nSUMMARY\n")

for col in ['sleep_hours','screen_time_hours','stress_level_0_10','productivity_0_100']:
    print(f"{col}: Mean={df[col].mean():.2f}, Std={df[col].std():.2f}")