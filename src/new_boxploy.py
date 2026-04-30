import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

df_lstm = pd.read_csv("model/results_lstm_ae.csv")
df_if = pd.read_csv("model/results_iforest.csv")

comp = df_lstm[["Dataset","Precision","Recall","F1"]].copy()
comp.columns = ["Dataset","Precision_LSTM","Recall_LSTM","F1_LSTM"]
comp["F1_IF"] = df_if["F1_IF"]
comp["Improvement"] = comp["F1_LSTM"] - comp["F1_IF"]
print(comp.to_string(index=False))
comp.to_csv("model/comparison.csv", index=False)

data = [df_lstm["F1"], df_if["F1_IF"]]
plt.figure(figsize=(6,4))
sns.boxplot(data=data)
plt.xticks([0,1], ["LSTM-AE", "Isolation Forest"])
plt.ylabel("F1-score")
plt.title("Сравнение методов на NAB")
plt.tight_layout()
plt.savefig("model/f1_comparison.png")