import pandas as pd


def main():
    print("Loading predictions and sample submission...")
    # Load predictions.csv
    # The predictions were saved using pd.DataFrame(preds).to_csv("output/predictions.csv", index=True)
    preds_df = pd.read_csv("output/predictions.csv", index_col=0)

    # Load sample submission
    sub_fl = pd.read_csv("data/sample_submission.csv", index_col="id")

    print(f"Predictions shape: {preds_df.shape}")
    print(f"Sample submission shape: {sub_fl.shape}")

    # Update target
    # The predictions are likely in the first column, which pandas might name '0'
    target_col = preds_df.columns[0]
    sub_fl["target"] = preds_df[target_col].values

    # Save submission
    out_path = "output/submission.csv"
    sub_fl.to_csv(out_path, index=True)

    print(f"Submission created successfully at {out_path}!")
    print(sub_fl.head())


if __name__ == "__main__":
    main()
