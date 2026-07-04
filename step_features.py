import pandas as pd

df = pd.read_csv("data/paysim_data.csv")

df = df[df["type"].isin(["TRANSFER", "CASH_OUT"])]

step_stats = (
    df.groupby("step")
      .agg(
          system_volume_this_step=("amount", "count"),
          step_avg_amt=("amount", "mean")
      )
      .reset_index()
)

step_stats.to_csv("models/step_stats.csv", index=False)

print(step_stats.head())