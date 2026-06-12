"""Generate EDA plots and save them as artifacts.
Creates histograms, correlation heatmap, and class-balance plot.
"""
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(__file__))
CSV = os.path.join(ROOT, "data", "processed_cleveland.csv")
OUT = os.path.join(ROOT, "artifacts", "eda")

def ensure():
    os.makedirs(OUT, exist_ok=True)

def run():
    ensure()
    df = pd.read_csv(CSV)

    # histograms for numeric columns
    num_cols = df.select_dtypes(include="number").columns.tolist()
    for c in num_cols:
        plt.figure()
        sns.histplot(df[c].dropna(), kde=False)
        plt.title(f"Distribution of {c}")
        fp = os.path.join(OUT, f"hist_{c}.png")
        plt.savefig(fp)
        plt.close()

    # correlation heatmap (numeric only)
    corr = df[num_cols].corr()
    plt.figure(figsize=(10,8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm")
    plt.title("Correlation matrix")
    plt.savefig(os.path.join(OUT, "correlation_heatmap.png"))
    plt.close()

    # class balance
    plt.figure()
    sns.countplot(x="target", data=df)
    plt.title("Target class balance")
    plt.savefig(os.path.join(OUT, "class_balance.png"))
    plt.close()

    print(f"EDA artifacts written to {OUT}")

if __name__ == "__main__":
    run()
