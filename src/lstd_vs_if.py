import pandas as pd

df_lstm = pd.read_csv("model/results_lstm_ae.csv")
df_if = pd.read_csv("model/results_iforest.csv")

comp = df_lstm[["Dataset","Precision","Recall","F1"]].copy()
comp.columns = ["Dataset","Precision_LSTM","Recall_LSTM","F1_LSTM"]
comp["F1_IF"] = df_if["F1_IF"]
comp["Improvement"] = comp["F1_LSTM"] - comp["F1_IF"]
print(comp.to_string(index=False))
comp.to_csv("model/comparison.csv", index=False)