# Monitoring

## Custom data drift checker

Since version 0.90 BlueCast checks for data drift for numerical
and categorical columns. The checks happen on the raw data.
Categories will be stored anonymized by default. Data drift
checks are not part of the model pipeline, but have to be called separately:

```sh
from bluecast.monitoring.data_monitoring import DataDrift


data_drift_checker = DataDrift()
# statistical data drift checks for numerical features
data_drift_checker.kolmogorov_smirnov_test(data, new_data, threshold=0.05)
# show flags
print(data_drift_checker.kolmogorov_smirnov_flags)

# statistical data drift checks for categorical features
data_drift_checker.population_stability_index(data, new_data)
# show flags
print(data_drift_checker.population_stability_index_flags)
# show psi values
print(data_drift_checker.population_stability_index_values)

# QQplot for two numerical columns
data_drift_checker.qqplot_two_samples(train["feature1"], test["feature1"], x_label="X", y_label="Y")
```

![QQplot example](docs/source/qqplot_sample.png)
