(1) training_curation.py: for every replicate (68 in total) in the training file, subtract every non-drug sample from the DHA sample and put the 24HR data after the 6HR data. (30 samples, 11080 features) Then standardize the features and centralize the label. 

(2) test_curation.py: Curate the test data just as how we deal with the training data. 

(3) parameter_determine.py: we first use the kpca to lower the dimension of the features and then use the sklearn_ridge to make the regression. Use the mse as the scoring function of cross_val_score to do the hyper-parameter tuning. 

(4) model_train_test.py: train the model and feed the model trained with the test data
