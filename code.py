import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from lifelines import KaplanMeierFitter
import warnings
warnings.filterwarnings("ignore")

print("="*70)
print("FAST PIPELINE ANALYSIS")
print("="*70)

print("\n[1/6] Loading...")
data = pd.read_excel("Merged_Pipeline_Data.xlsx").dropna()
print(f"Full: {data.shape}")
data = data.sample(n=50000, random_state=42).reset_index(drop=True)
print(f"Sample: {data.shape}")

X = StandardScaler().fit_transform(data.select_dtypes(include=[np.number]).values)

print("\n[2/6] Isolation Forest...")
iso = IsolationForest(contamination=0.1, random_state=42, n_estimators=100)
iso.fit(X)
data["anom"] = (iso.predict(X) == -1).astype(int)
data["score"] = iso.decision_function(X)
print(f"Anomalies: {data['anom'].sum()}")

print("\n[3/6] Classification...")
p = data["score"].quantile([0.25, 0.5, 0.75])
data["sev"] = data["score"].apply(lambda s: "Major" if s<p[0.25] else "Minor" if s<p[0.5] else "Micro" if s<p[0.75] else "Normal")
print(data["sev"].value_counts())

print("\n[4/6] Clustering...")
km = KMeans(n_clusters=3, random_state=42, n_init=10)
data["risk"] = km.fit_predict(X)
data["risk_lbl"] = data["risk"].map({0:"Low", 1:"Med", 2:"High"})
print(data["risk_lbl"].value_counts())

print("\n[5/6] Survival...")
data["t"] = range(1, len(data)+1)
kmf = KaplanMeierFitter()
kmf.fit(data["t"], data["anom"])
med = kmf.median_survival_time_
print(f"Median: {med:.1f}")

print("\n[6/6] Plotting...")
fig1, ax1 = plt.subplots(2, 3, figsize=(18, 10))
fig1.suptitle("Pipeline Analysis Dashboard", fontsize=16, fontweight="bold")

ax1[0,0].hist(data["score"], bins=50, color="steelblue", edgecolor="black")
ax1[0,0].set_title("Anomaly Scores")

ax1[0,1].scatter(range(len(data)), data.iloc[:, 0], c=data["anom"], cmap="RdYlGn_r", s=10, alpha=0.6)
ax1[0,1].set_title("Signal with Anomalies")

sev = data["sev"].value_counts()
ax1[0,2].bar(sev.index, sev.values, color=["red","orange","yellow","green"], edgecolor="black")
ax1[0,2].set_title("Severity")
ax1[0,2].tick_params(axis="x", rotation=45)

risk = data["risk_lbl"].value_counts()
ax1[1,0].bar(risk.index, risk.values, color=["green","yellow","red"], edgecolor="black")
ax1[1,0].set_title("Risk")

ax1[1,1].scatter(X[:, 0], X[:, min(1,X.shape[1]-1)], c=data["risk"], cmap="viridis", s=20, alpha=0.6)
ax1[1,1].set_title("Clustering")

kmf.plot_survival_function(ax=ax1[1,2], linewidth=3, color="steelblue")
ax1[1,2].axvline(med, color="red", linestyle="--", label=f"Median: {med:.1f}")
ax1[1,2].set_title("Survival")
ax1[1,2].legend()

plt.tight_layout()
plt.savefig("results.png", dpi=300)
plt.show()

data.to_csv("results.csv", index=False)

print("\n" + "="*70)
print("DONE!")
print(f"Samples: {len(data)}")
print(f"Anomalies: {data['anom'].sum()} ({data['anom'].sum()/len(data)*100:.1f}%)")
print("Files: results.png, results.csv")
print("="*70)
