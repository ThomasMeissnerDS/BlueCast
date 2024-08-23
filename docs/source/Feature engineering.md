# Feature engineering

A critical part of machine learning is feature engineering.
BlueCast's pipelines will automatically execute only
necessary feature engineering and leaves this to the end
user. However BlueCast offers some tools for feature
engineering to make this part more approachable and
faster.

First we import the required modules:

```sh
from bluecast.preprocessing.feature_types import FeatureTypeDetector
from bluecast.preprocessing.feature_creation import AddRowLevelAggFeatures, GroupLevelAggFeatures
```

Next we can make use of `FeatureTypeDetector` to identify
numerical columns:

```sh
ignore_cols = [TARGET, "id", "CustomerId"]

feat_type_detector = FeatureTypeDetector()
train_data = feat_type_detector.fit_transform_feature_types(train.drop(ignore_cols, axis=1))
```

Next we use `AddRowLevelAggFeatures` to create features
on row level. This usually adds a small degree of
additional performance.

```sh
agg_feat_creator = AddRowLevelAggFeatures()

train_num = agg_feat_creator.add_row_level_agg_features(train.loc[:, feat_type_detector.num_columns])
test_num = agg_feat_creator.add_row_level_agg_features(test.loc[:, feat_type_detector.num_columns])

train_num = train_num.drop(agg_feat_creator.original_features, axis=1)
test_num = test_num.drop(agg_feat_creator.original_features, axis=1)


train = pd.concat([train, train_num], axis=1)
test = pd.concat([test, test_num], axis=1)
```

Additionally we can also provide information via group
aggregations with `GroupLevelAggFeatures`:

```python
group_agg_creator = GroupLevelAggFeatures()

train_num = group_agg_creator.create_groupby_agg_features(
    df = train,
    groupby_columns=["Geography", "Gender", "NumOfProducts"],
    columns_to_agg=feat_type_detector.num_columns, # None = take all
    target_col=None,
    aggregations = None # falls back to some aggs
)

test_num = group_agg_creator.create_groupby_agg_features(
    df = test,
    groupby_columns=["Geography", "Gender", "NumOfProducts"],
    columns_to_agg=feat_type_detector.num_columns, # None = take all
    target_col=TARGET,
    aggregations = None # falls back to some aggs
)

# joining the train information everywhere
train = train.merge(train_num, on=["Geography", "Gender", "NumOfProducts"], how="left")
test = test.merge(train_num, on=["Geography", "Gender", "NumOfProducts"], how="left")
```

Please note that this will increase the number of features
significantly.
