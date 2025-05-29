# dpart

## Official repository
https://github.com/hazy/dpart

## License
MIT License

## Changes
* Local imports paths.
* Modify the differential private synthpop class to include more arguments.
* Implement the privacy budget accountant.
* Replace the lambda function in the dpart class by a named function so that the object can be saved with pickle.
* Change feature_range of the MinMaxScaler from list to tuple to match the type required by sklearn.
* Convert the type of the columns to str type when generating synthetic data.
* Update maximum number of iterations for logistic regression to fix the convergence issue.
* Set the entropy to its one-sided limitation when the probability is 0 to avoid returning NA.
* Remove warnings when methods to generate each variable are not specified.
* Remove warning from DP-models.
* Fix typo in warning message for privacy leakage of categorical variables.
* Fix unshown warning message for the bounds of continuous variables.
* Fix privacy leakage for decoding predictions.
* Fix hardcoded bin number in bin_encoder.
* Used tree model as default model.
