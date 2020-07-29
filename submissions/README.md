# Note
the submission files will be here, since CSV, they will not be uploaded.


0.967159 and 0.972724. ---------------- valid_0's rmse: 0.918846 with 300 boosting rounds
0.990274 and 0.993795. ---------------- valid_0's rmse: 0.83095 (0.89 ?? ) with 300 boosting rounds
0.994544 and 0.997140. ---------------- valid_0's rmse: 0.853319 (0.91 ?? ) with 101 boosting rounds (some featues causing overfit)
1.001826 and 1.001773. ---------------- valid_0's rmse: 0.909856 (0.95 ?? ) with 150 
0.939209 and 0.943882. ---------------- [201]	valid_0's rmse: 0.911661 (0.96 ?) with 300 [28jul2020_lgb_all_feat_12.csv]
0.935320 and 0.940645. ---------------- [297]	valid_0's rmse: 0.910048 (0.91 ?) with 300.. removed max_bin, feature_fraction : 0.5 > 0.75 [28jul2020_lgb_all_feat_13.csv]

changing:
'num_leaves': 71,>> 2**7
'min_data_in_leaf': 71, >> 2**7
next will change 'bagging_fraction': 0.5,  >> 0.75 (not this try) and also remove the 'extra_trees' 

0.933745 and 0.939378. ---------------- [185]	valid_0's rmse: 0.903785 (0.91 ?) [28jul2020_lgb_all_feat_14.csv] best....


['shop_id', 'item_id', 'item_category_id', 'name_2', 'name_3',
       'num_days', 'revenue_lag_1', 'target_lag_1', 'target_item_lag_1',
       'target_shop_lag_1', 'target_lag_2', 'target_item_lag_2',
       'target_shop_lag_2', 'target_lag_3', 'target_item_lag_3',
       'target_shop_lag_3', 'target_lag_12', 'target_item_lag_12',
       'target_shop_lag_12', 'month_mean_target', 'parent_cat_db_mean_target',
       'shop_city_mean', 'db_target_mean']


changing:
'lambda_l1': 0.001,
'lambda_l2': 0.001,
'extra_trees': True
'bagging_fraction': 0.5 >> 0.75, 
0.935052 and 0.941154. ---------------- [233]	valid_0's rmse: 0.902384 (0.92 ?) [28jul2020_lgb_all_feat_15.csv]

changing:
adding back only lambdas
removing 'target_lag_12', 'target_item_lag_12','target_shop_lag_12'
0.951058 and 0.954563. ---------------- [190]	valid_0's rmse: 0.911352 (0.94?) [28jul2020_lgb_all_feat_16.csv]


IMPORTANT: 
change to be added: same configuration, but with lag_4, and means (both db and non db for shop_id, item_id and others)

# Submissions on 29-Jul-2020 # 1

## Kaggle

### Changes 
- Added Lag_4 for target, target_shop & target_item
- Added shop_id_mean_target

### Fit Columns

Total 29 as follows: 

```
'shop_id', 'item_id', 'item_category_id', 'name_2', 'name_3',
'num_days', 'revenue_lag_1', 'target_lag_1', 'target_item_lag_1',
'target_shop_lag_1', 'target_lag_2', 'target_item_lag_2',
'target_shop_lag_2', 'target_lag_3', 'target_item_lag_3',
'target_shop_lag_3', 'target_lag_4', 'target_item_lag_4',
'target_shop_lag_4', 'target_lag_12', 'target_item_lag_12',
'target_shop_lag_12', 'month_mean_target', 'parent_cat_id_mean_target',
'parent_cat_db_mean_target', 'item_cat_mean_target',
'shop_id_mean_target', 'shop_city_mean', 'db_target_mean'
```



```
lgb_params = {
               'feature_fraction': 0.75,
               'metric': 'rmse',
               'nthread':-1, 
               'min_data_in_leaf': 2**7, 
               'bagging_fraction': 0.75, 
               'learning_rate': 0.03, 
               'objective': 'mse', 
               'bagging_seed': 2**7, 
               'num_leaves': 2**7,
               'bagging_freq':1,
               'verbose':1,
               'lambda_l1': 0.001,
               'lambda_l2': 0.001
              }

model = lgb.train(lgb_params, 
                  lgb.Dataset(X_train,label=y_train), 
                  300 , #categorical_feature=cat_features, 
                  valid_sets=lgb.Dataset(X_valid,label=y_valid), verbose_eval=True,
                 early_stopping_rounds=15)

```

### Results on Kaggle:

```
File: 29jul2020_lgb_all_feat_17.csv
Early stopping, best iteration is: [225]	valid_0's rmse: 0.902938
Kaggle: 0.91481 (Best Score ... top 24%)
```


# Submissions on 29-Jul-2020 # 2

### Changes 
added an attribute: shop_id_exp_y_mean
rest remains the same as above

### Results on Kaggle:
```
File: 29jul2020_lgb_all_feat_18.csv
Early stopping, best iteration is: [269]	valid_0's rmse: 0.903695
Kaggle: 0.92030
```

# Submissions on 29-Jul-2020 # 3

### Changes 
No changes, even the model is not trained this time, except using the predictions for 
29jul2020_lgb_all_feat_18.csv and 29jul2020_lgb_all_feat_17.csv
and averaging them to see if this improves the results. 

```
File: 29jul2020_lgb_all_feat_17_18_avg.csv
Kaggle: 0.91675
```