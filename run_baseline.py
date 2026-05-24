import pandas as pd

from bluecast.blueprints.cast_regression import BlueCastRegression

train = pd.read_csv("data/dataset.csv")
test = pd.read_csv("data/test.csv")
sub = pd.read_csv("data/sample_submission.csv")

bc = BlueCastRegression(class_problem="regression")
bc.fit(train, target_col="target")
preds = bc.predict(test)
sub["target"] = preds
sub.to_csv("baseline_sub.csv", index=False)
print("Done!")
