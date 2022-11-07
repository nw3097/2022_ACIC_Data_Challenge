import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn import metrics, preprocessing
import lightgbm as lgb
from sklearn.model_selection import train_test_split, cross_val_predict
from scipy.stats import norm

# propensity score features for lgbm propensity score model
ps_features = ['Y_mean_pre','q_1','q_2','q_3','q_4','q_5','d','year1_only','year2_only','both_year','num_patients',\
'V1_mean','V1_sd','V2_mean','V2_sd','V3_mean','V3_sd','V4_mean','V4_sd','V5_A_mean','V5_B_mean','V5_C_mean',\
'X1','X2_int_coded','X3','X4_int_coded','X5','X6','X7','X8','X9']
cat_features=['X2_int_coded','X4_int_coded']

# features for lgbm nuisance models
all_features_lgb = ['Y_mean_pre','q_1','q_2','q_3','q_4','q_5','d','year1_only','year2_only','both_year','num_patients',\
'V1_mean','V1_sd','V2_mean','V2_sd','V3_mean','V3_sd','V4_mean','V4_sd','V5_A_mean','V5_B_mean','V5_C_mean',\
'X1','C(X2_int_coded)','X3','C(X4_int_coded)','X5','X6','X7','X8','X9','V1','V2','V3','V4','C(V5_int_coded)','Y1','Y2','C(year)']
all_features_lgb_without_C = [feature[2:-1] if feature[:2] == 'C(' else feature for feature in all_features_lgb]
cat_features_lgb = [feature[2:-1] for feature in all_features_lgb if feature[:2] == 'C(']

# features for wls
ps_features_for_reg = ps_features.copy()
for i in range(len(ps_features_for_reg)):
    if ps_features_for_reg[i] == 'X2_int_coded':
        ps_features_for_reg[i] = 'C(X2)'
    if ps_features_for_reg[i] == 'X4_int_coded':
        ps_features_for_reg[i] = 'C(X4)'
V_features = ['V1','V2','V3','V4','C(V5)']
time_features = ['C(year)']
pre_spending_features = ['Y1','Y2']

def load_data(seq_no):
    """
    seq_no: sequence number of dataset, ranging from 1 to 3400
    load four raw datasets and one dataframe that combines them
    """
    if seq_no <= 1200:
        file_path = "/Users/jx482q/acic_2022_data_challenge/data/track1a"
    elif seq_no <= 2400:
        file_path = "/Users/jx482q/acic_2022_data_challenge/data/track1b"
    else:
        file_path = "/Users/jx482q/acic_2022_data_challenge/data/track1c"
    padded_seq_no = f'{seq_no:04}'    
        
    df1 = pd.read_csv(file_path+"/patient/acic_patient_"+padded_seq_no+".csv")
    df2 = pd.read_csv(file_path+"/patient_year/acic_patient_year_"+padded_seq_no+".csv")
    df3 = pd.read_csv(file_path+"/practice/acic_practice_"+padded_seq_no+".csv")
    df4 = pd.read_csv(file_path+"/practice_year/acic_practice_year_"+padded_seq_no+".csv")

    df = df1.merge(df2, on='id.patient')
    df = df.merge(df3, on='id.practice')
    df4 = df4.rename(columns={'Y':'Y_avg'})
    df = df.merge(df4, on=['id.practice','year'])

    return (df1, df2, df3, df4, df)

def build_ps_df(df3, df4, df, quantile_granularity=5):
    """
    df3: practice dataframe
    df4: practice year dataframe
    df: dataframe that combines the four raw datasets
    quantile_granularity: e.g. 5 means (0.2, 0.4, 0.6, 0.8, 1.0)
    return: dataframe that has all features needed for propensity score estimation
    """
    # 1. mean spending in pre period
    tmp1 = df[['id.practice','post','Y']].groupby(['id.practice','post']).mean().reset_index()
    tmp1 = tmp1[tmp1.post==0].reset_index(drop=True)
    tmp1 = tmp1.rename(columns={'Y':'Y_mean_pre'})
    tmp1 = tmp1.drop(columns='post')

    # 2. spending quantiles in pre period
    q = [(1/quantile_granularity)*x for x in range(1,quantile_granularity+1)]
    quantile_col_names = ['q_'+str(i) for i in range(1, quantile_granularity+1)]

    tmp2 = df[['id.practice','post','Y']].groupby(['id.practice','post']).quantile(q).reset_index()
    tmp2 = tmp2.pivot(index=['id.practice','post'], columns='level_2', values='Y').reset_index()
    tmp2.columns = ['id.practice','post'] + quantile_col_names
    tmp2 = tmp2[tmp2.post==0].reset_index(drop=True)
    tmp2 = tmp2.drop(columns='post')

    # 3. per capita spending growth from year 1 to year 2
    tmp3 = df4[df4.post==0].reset_index(drop=True)
    tmp3 = tmp3[['id.practice','year','Y_avg']]
    tmp3 = tmp3.pivot(index='id.practice', columns='year', values='Y_avg').reset_index()
    tmp3.columns = ['id.practice','Y1','Y2']
    tmp3['d'] = tmp3.Y2 - tmp3.Y1
    tmp3 = tmp3.drop(columns=['Y1','Y2'])

    # 4. share of patients with both year 1 and year 2 data, year 1 only, year 2 only
    tmp4 = df[df.post==0][['id.patient','id.practice','year']].reset_index(drop=True)
    tmp4 = tmp4[['id.practice','id.patient','year']].groupby(['id.practice','id.patient']).sum().reset_index()
    tmp4 = tmp4.groupby(['id.practice','year']).count().reset_index()
    tmp4 = tmp4.pivot(index='id.practice',columns='year',values='id.patient').reset_index()
    tmp4.columns = ['id.practice','year1_only','year2_only','both_year']
    tmp4 = tmp4.fillna(0)
    tmp4['num_patients'] = tmp4.year1_only + tmp4.year2_only + tmp4.both_year
    tmp4['year1_only'] = tmp4.year1_only/tmp4.num_patients
    tmp4['year2_only'] = tmp4.year2_only/tmp4.num_patients
    tmp4['both_year'] = tmp4.both_year/tmp4.num_patients

    # 5. mean and standard deviation of patient features in pre period
    tmp5 = df[df.post==0][['id.practice','id.patient','post','V1','V2','V3','V4','V5']].drop_duplicates().reset_index(drop=True)
    tmp5['V5_A'] = (tmp5.V5=='A').astype(int)
    tmp5['V5_B'] = (tmp5.V5=='B').astype(int)
    tmp5['V5_C'] = (tmp5.V5=='C').astype(int)
    tmp5 = tmp5.drop(columns=['id.patient','post','V5'])
    f = {'V1':['mean','std'],'V2':['mean','std'],'V3':['mean','std'],'V4':['mean','std'],'V5_A':'mean','V5_B':'mean','V5_C':'mean'}
    tmp5 = tmp5.groupby('id.practice').agg(f).reset_index()
    tmp5.columns = ['id.practice','V1_mean','V1_sd','V2_mean','V2_sd','V3_mean','V3_sd','V4_mean','V4_sd','V5_A_mean','V5_B_mean','V5_C_mean']

    # get treatment assignment
    tmp6=df4[['id.practice','Z']].drop_duplicates().reset_index()
    df3_PS = df3.merge(tmp6, on='id.practice')

    # encode categorical features into integer features
    cat_features=['X2','X4']
    le = preprocessing.LabelEncoder()
    for col in cat_features:
        unique_cats = np.array(list(set(df3_PS[col])))
        le.fit(unique_cats)
        df3_PS[col+'_int_coded'] = le.transform(df3_PS[col])

    # add more features
    df3_PS = df3_PS.merge(tmp1, on='id.practice')
    df3_PS = df3_PS.merge(tmp2, on='id.practice')
    df3_PS = df3_PS.merge(tmp3, on='id.practice')
    df3_PS = df3_PS.merge(tmp4, on='id.practice')
    df3_PS = df3_PS.merge(tmp5, on='id.practice')

    return df3_PS

def do_lgbm_ps_model(data, all_features, cat_features, label_col):
    """
    data: data used in ps modeling
    all_features: list of features used in ps model
    cat_features: categorical features used in ps model
    label_col: label column
    return: lgbm model object
    """

    # set up train test data
    df_train, df_test = train_test_split(data, test_size=0.3, random_state=1)
    y_train = df_train[label_col]
    y_test = df_test[label_col]
    X_train = df_train[all_features]
    X_test = df_test[all_features]

    lgb_train = lgb.Dataset(X_train, 
                            label=y_train,
                            feature_name=all_features,
                            categorical_feature=cat_features)
    lgb_eval = lgb.Dataset(X_test, 
                           label=y_test, 
                           feature_name=all_features,
                           categorical_feature=cat_features,
                           reference=lgb_train)
    params = {
    'boosting_type':'gbdt',
    'objective':'binary',
    'min_data_in_leaf':60,
    'feature_fraction':0.7
    }

    gbm = lgb.train(params,
                    lgb_train,
                    num_boost_round=100,
                    valid_sets=lgb_eval,
                    early_stopping_rounds=10)

    return gbm

def add_ps_weights(data, model, ps_features):
    """
    data: data for propensity score modeling
    model: propensity score model
    ps_features: features used in propensity score modeling
    return: add ps weights to data
    """
    data['ps'] = model.predict(data[ps_features], num_iteration=model.best_iteration)    

    # drop the control practices that have ps less than the lowest ps of treated practices
    lb_T = data[data.Z==1].ps.min()
    data = data[data.ps>=lb_T].reset_index(drop=True)

    # define weights
    data['w'] = data.Z + (1-data.Z)*(data.ps/(1-data.ps))
    
    return data

def setup_estimation_data(main_data, ps_data):
    """
    main_data: patient year level data
    ps_data: practice level data with ps features
    return: estimation data
    """

    df_pre = main_data[main_data.post==0].reset_index(drop=True)
    df_post = main_data[main_data.post==1].reset_index(drop=True)

    df_pre = df_pre[['id.patient','year','Y']]
    df_pre = pd.pivot(df_pre, index='id.patient',columns='year',values='Y').reset_index()
    df_pre.columns = ['id.patient','Y1','Y2']
    df_pre['pre_trend'] = df_pre.Y2 - df_pre.Y1
    df_post = df_post[['id.patient','id.practice','year','Y','V1','V2','V3','V4','V5']]
    est_data = df_post.merge(df_pre, how='left', on='id.patient')
    est_data = est_data.merge(ps_data, on='id.practice')

    return est_data
    
def do_dml_full_pipeline_with_weighting(reg_data):
    """
    reg_data: data used to do double machine learning
    return: data frame containing all estimates for a given data set
    """
    le = preprocessing.LabelEncoder()
    le.fit(np.array(list(set(reg_data['V5']))))
    reg_data['V5_int_coded'] = le.transform(reg_data['V5'])
    
    # 7. nuisance stage: outcome ~ controls (use weights in 2)
    model = lgb.LGBMRegressor(num_boost_round=30, random_state=1)
    Y_hat = cross_val_predict(model, reg_data[all_features_lgb_without_C], reg_data['Y'], 
                              fit_params={
                                  'categorical_feature': cat_features_lgb,
                                  'sample_weight': reg_data['w'],
                              },)    

    # 8. nuisance stage: treatment ~ controls (use weights in 2)
    # treatment = Z
    # controls: right hand side of reg_fmla except for Z
    model = lgb.LGBMClassifier(num_boost_round=30, random_state=1)
    Z_hat = cross_val_predict(model, reg_data[all_features_lgb_without_C], reg_data['Z'], 
                              fit_params={
                                  'categorical_feature': cat_features_lgb,
                                  'sample_weight': reg_data['w'],
                              },method='predict_proba')[:,1]

    # 9. target stage: residualized outcome ~ residualized treatment (used weights in 2)
    # (Y - Y_hat) ~ (Z - Z_hat), where the hats are predicted values from 7 and 8
    # https://www.statsmodels.org/devel/examples/notebooks/generated/wls.html
    Y_residual = reg_data['Y'] - Y_hat
    Z_residual = reg_data['Z'] - Z_hat

    confidence_level = 0.90
    res_df = pd.DataFrame()

    # do overall satt
    small_df = pd.DataFrame()
    reg_fmla = 'Y_residual ~ -1 + Z_residual'
    res = smf.wls(formula = reg_fmla, data=reg_data, weights=reg_data['w']).fit(cov_type='cluster',
            cov_kwds={'groups': reg_data['id.practice']})
    small_df['variable'] = pd.Series('Overall')
    small_df['level'] = pd.Series('NA')
    small_df['year'] = pd.Series('NA')
    small_df['satt'] = pd.Series(res.params['Z_residual'])
    small_df['lower90'] = pd.Series(res.params['Z_residual'] - norm.ppf(1 - (1 - confidence_level)/2)*res.bse['Z_residual'])
    small_df['upper90'] = pd.Series(res.params['Z_residual'] + norm.ppf(1 - (1 - confidence_level)/2)*res.bse['Z_residual'])

    res_df = res_df.append(small_df)

    # do by year satt
    small_df_3 = pd.DataFrame()
    small_df_4 = pd.DataFrame()
    reg_fmla = 'Y_residual ~ C(year):Z_residual'
    res = smf.wls(formula = reg_fmla, data=reg_data, weights=reg_data['w']).fit(cov_type='cluster',
            cov_kwds={'groups': reg_data['id.practice']})

    small_df_3['variable'] = pd.Series('Overall')
    small_df_3['level'] = pd.Series('NA')
    small_df_3['year'] = pd.Series(3)
    small_df_3['satt'] = pd.Series(res.params['C(year)[3]:Z_residual'])
    small_df_3['lower90'] = pd.Series(res.params['C(year)[3]:Z_residual'] - norm.ppf(1 - (1 - confidence_level)/2)*res.bse['C(year)[3]:Z_residual'])
    small_df_3['upper90'] = pd.Series(res.params['C(year)[3]:Z_residual'] + norm.ppf(1 - (1 - confidence_level)/2)*res.bse['C(year)[3]:Z_residual'])

    small_df_4['variable'] = pd.Series('Overall')
    small_df_4['level'] = pd.Series('NA')
    small_df_4['year'] = pd.Series(4)
    small_df_4['satt'] = pd.Series(res.params['C(year)[4]:Z_residual'])
    small_df_4['lower90'] = pd.Series(res.params['C(year)[4]:Z_residual'] - norm.ppf(1 - (1 - confidence_level)/2)*res.bse['C(year)[4]:Z_residual'])
    small_df_4['upper90'] = pd.Series(res.params['C(year)[4]:Z_residual'] + norm.ppf(1 - (1 - confidence_level)/2)*res.bse['C(year)[4]:Z_residual'])

    res_df = res_df.append(small_df_3)
    res_df = res_df.append(small_df_4)

    # do group satt
    # X1
    small_df_X1_0 = pd.DataFrame()
    small_df_X1_1 = pd.DataFrame()

    reg_fmla = 'Y_residual ~ C(X1):Z_residual'
    res = smf.wls(formula = reg_fmla, data=reg_data, weights=reg_data['w']).fit(cov_type='cluster',
            cov_kwds={'groups': reg_data['id.practice']})

    small_df_X1_0['variable'] = pd.Series('X1')
    small_df_X1_0['level'] = pd.Series('0')
    small_df_X1_0['year'] = pd.Series('NA')
    small_df_X1_0['satt'] = pd.Series(res.params['C(X1)[0]:Z_residual'])
    small_df_X1_0['lower90'] = pd.Series(res.params['C(X1)[0]:Z_residual'] - norm.ppf(1 - (1 - confidence_level)/2)*res.bse['C(X1)[0]:Z_residual'])
    small_df_X1_0['upper90'] = pd.Series(res.params['C(X1)[0]:Z_residual'] + norm.ppf(1 - (1 - confidence_level)/2)*res.bse['C(X1)[0]:Z_residual'])

    small_df_X1_1['variable'] = pd.Series('X1')
    small_df_X1_1['level'] = pd.Series('1')
    small_df_X1_1['year'] = pd.Series('NA')
    small_df_X1_1['satt'] = pd.Series(res.params['C(X1)[1]:Z_residual'])
    small_df_X1_1['lower90'] = pd.Series(res.params['C(X1)[1]:Z_residual'] - norm.ppf(1 - (1 - confidence_level)/2)*res.bse['C(X1)[1]:Z_residual'])
    small_df_X1_1['upper90'] = pd.Series(res.params['C(X1)[1]:Z_residual'] + norm.ppf(1 - (1 - confidence_level)/2)*res.bse['C(X1)[1]:Z_residual'])

    res_df = res_df.append(small_df_X1_0)
    res_df = res_df.append(small_df_X1_1)

    # do group satt
    # X2
    small_df_X2_A = pd.DataFrame()
    small_df_X2_B = pd.DataFrame()
    small_df_X2_C = pd.DataFrame()

    reg_fmla = 'Y_residual ~ C(X2):Z_residual'
    res = smf.wls(formula = reg_fmla, data=reg_data, weights=reg_data['w']).fit(cov_type='cluster',
            cov_kwds={'groups': reg_data['id.practice']})

    small_df_X2_A['variable'] = pd.Series('X2')
    small_df_X2_A['level'] = pd.Series('A')
    small_df_X2_A['year'] = pd.Series('NA')
    small_df_X2_A['satt'] = pd.Series(res.params['C(X2)[A]:Z_residual'])
    small_df_X2_A['lower90'] = pd.Series(res.params['C(X2)[A]:Z_residual'] - norm.ppf(1 - (1 - confidence_level)/2)*res.bse['C(X2)[A]:Z_residual'])
    small_df_X2_A['upper90'] = pd.Series(res.params['C(X2)[A]:Z_residual'] + norm.ppf(1 - (1 - confidence_level)/2)*res.bse['C(X2)[A]:Z_residual'])

    small_df_X2_B['variable'] = pd.Series('X2')
    small_df_X2_B['level'] = pd.Series('B')
    small_df_X2_B['year'] = pd.Series('NA')
    small_df_X2_B['satt'] = pd.Series(res.params['C(X2)[B]:Z_residual'])
    small_df_X2_B['lower90'] = pd.Series(res.params['C(X2)[B]:Z_residual'] - norm.ppf(1 - (1 - confidence_level)/2)*res.bse['C(X2)[B]:Z_residual'])
    small_df_X2_B['upper90'] = pd.Series(res.params['C(X2)[B]:Z_residual'] + norm.ppf(1 - (1 - confidence_level)/2)*res.bse['C(X2)[B]:Z_residual'])

    small_df_X2_C['variable'] = pd.Series('X2')
    small_df_X2_C['level'] = pd.Series('C')
    small_df_X2_C['year'] = pd.Series('NA')
    small_df_X2_C['satt'] = pd.Series(res.params['C(X2)[C]:Z_residual'])
    small_df_X2_C['lower90'] = pd.Series(res.params['C(X2)[C]:Z_residual'] - norm.ppf(1 - (1 - confidence_level)/2)*res.bse['C(X2)[C]:Z_residual'])
    small_df_X2_C['upper90'] = pd.Series(res.params['C(X2)[C]:Z_residual'] + norm.ppf(1 - (1 - confidence_level)/2)*res.bse['C(X2)[C]:Z_residual'])

    res_df = res_df.append(small_df_X2_A)
    res_df = res_df.append(small_df_X2_B)
    res_df = res_df.append(small_df_X2_C)

    # do group satt
    # X3
    small_df_X3_0 = pd.DataFrame()
    small_df_X3_1 = pd.DataFrame()

    reg_fmla = 'Y_residual ~ C(X3):Z_residual'
    res = smf.wls(formula = reg_fmla, data=reg_data, weights=reg_data['w']).fit(cov_type='cluster',
            cov_kwds={'groups': reg_data['id.practice']})

    small_df_X3_0['variable'] = pd.Series('X3')
    small_df_X3_0['level'] = pd.Series('0')
    small_df_X3_0['year'] = pd.Series('NA')
    small_df_X3_0['satt'] = pd.Series(res.params['C(X3)[0]:Z_residual'])
    small_df_X3_0['lower90'] = pd.Series(res.params['C(X3)[0]:Z_residual'] - norm.ppf(1 - (1 - confidence_level)/2)*res.bse['C(X3)[0]:Z_residual'])
    small_df_X3_0['upper90'] = pd.Series(res.params['C(X3)[0]:Z_residual'] + norm.ppf(1 - (1 - confidence_level)/2)*res.bse['C(X3)[0]:Z_residual'])

    small_df_X3_1['variable'] = pd.Series('X3')
    small_df_X3_1['level'] = pd.Series('1')
    small_df_X3_1['year'] = pd.Series('NA')
    small_df_X3_1['satt'] = pd.Series(res.params['C(X3)[1]:Z_residual'])
    small_df_X3_1['lower90'] = pd.Series(res.params['C(X3)[1]:Z_residual'] - norm.ppf(1 - (1 - confidence_level)/2)*res.bse['C(X3)[1]:Z_residual'])
    small_df_X3_1['upper90'] = pd.Series(res.params['C(X3)[1]:Z_residual'] + norm.ppf(1 - (1 - confidence_level)/2)*res.bse['C(X3)[1]:Z_residual'])

    res_df = res_df.append(small_df_X3_0)
    res_df = res_df.append(small_df_X3_1)

    # do group satt
    # X4
    small_df_X4_A = pd.DataFrame()
    small_df_X4_B = pd.DataFrame()
    small_df_X4_C = pd.DataFrame()

    reg_fmla = 'Y_residual ~ C(X4):Z_residual'
    res = smf.wls(formula = reg_fmla, data=reg_data, weights=reg_data['w']).fit(cov_type='cluster',
            cov_kwds={'groups': reg_data['id.practice']})

    small_df_X4_A['variable'] = pd.Series('X4')
    small_df_X4_A['level'] = pd.Series('A')
    small_df_X4_A['year'] = pd.Series('NA')
    small_df_X4_A['satt'] = pd.Series(res.params['C(X4)[A]:Z_residual'])
    small_df_X4_A['lower90'] = pd.Series(res.params['C(X4)[A]:Z_residual'] - norm.ppf(1 - (1 - confidence_level)/2)*res.bse['C(X4)[A]:Z_residual'])
    small_df_X4_A['upper90'] = pd.Series(res.params['C(X4)[A]:Z_residual'] + norm.ppf(1 - (1 - confidence_level)/2)*res.bse['C(X4)[A]:Z_residual'])

    small_df_X4_B['variable'] = pd.Series('X4')
    small_df_X4_B['level'] = pd.Series('B')
    small_df_X4_B['year'] = pd.Series('NA')
    small_df_X4_B['satt'] = pd.Series(res.params['C(X4)[B]:Z_residual'])
    small_df_X4_B['lower90'] = pd.Series(res.params['C(X4)[B]:Z_residual'] - norm.ppf(1 - (1 - confidence_level)/2)*res.bse['C(X4)[B]:Z_residual'])
    small_df_X4_B['upper90'] = pd.Series(res.params['C(X4)[B]:Z_residual'] + norm.ppf(1 - (1 - confidence_level)/2)*res.bse['C(X4)[B]:Z_residual'])

    small_df_X4_C['variable'] = pd.Series('X4')
    small_df_X4_C['level'] = pd.Series('C')
    small_df_X4_C['year'] = pd.Series('NA')
    small_df_X4_C['satt'] = pd.Series(res.params['C(X4)[C]:Z_residual'])
    small_df_X4_C['lower90'] = pd.Series(res.params['C(X4)[C]:Z_residual'] - norm.ppf(1 - (1 - confidence_level)/2)*res.bse['C(X4)[C]:Z_residual'])
    small_df_X4_C['upper90'] = pd.Series(res.params['C(X4)[C]:Z_residual'] + norm.ppf(1 - (1 - confidence_level)/2)*res.bse['C(X4)[C]:Z_residual'])

    res_df = res_df.append(small_df_X4_A)
    res_df = res_df.append(small_df_X4_B)
    res_df = res_df.append(small_df_X4_C)

    # do group satt
    # X5
    small_df_X5_0 = pd.DataFrame()
    small_df_X5_1 = pd.DataFrame()

    reg_fmla = 'Y_residual ~ C(X5):Z_residual'
    res = smf.wls(formula = reg_fmla, data=reg_data, weights=reg_data['w']).fit(cov_type='cluster',
            cov_kwds={'groups': reg_data['id.practice']})

    small_df_X5_0['variable'] = pd.Series('X5')
    small_df_X5_0['level'] = pd.Series('0')
    small_df_X5_0['year'] = pd.Series('NA')
    small_df_X5_0['satt'] = pd.Series(res.params['C(X5)[0]:Z_residual'])
    small_df_X5_0['lower90'] = pd.Series(res.params['C(X5)[0]:Z_residual'] - norm.ppf(1 - (1 - confidence_level)/2)*res.bse['C(X5)[0]:Z_residual'])
    small_df_X5_0['upper90'] = pd.Series(res.params['C(X5)[0]:Z_residual'] + norm.ppf(1 - (1 - confidence_level)/2)*res.bse['C(X5)[0]:Z_residual'])

    small_df_X5_1['variable'] = pd.Series('X5')
    small_df_X5_1['level'] = pd.Series('1')
    small_df_X5_1['year'] = pd.Series('NA')
    small_df_X5_1['satt'] = pd.Series(res.params['C(X5)[1]:Z_residual'])
    small_df_X5_1['lower90'] = pd.Series(res.params['C(X5)[1]:Z_residual'] - norm.ppf(1 - (1 - confidence_level)/2)*res.bse['C(X5)[1]:Z_residual'])
    small_df_X5_1['upper90'] = pd.Series(res.params['C(X5)[1]:Z_residual'] + norm.ppf(1 - (1 - confidence_level)/2)*res.bse['C(X5)[1]:Z_residual'])

    res_df = res_df.append(small_df_X5_0)
    res_df = res_df.append(small_df_X5_1)
    return res_df

def do_dml_full_pipeline_without_weighting(reg_data):
    """
    reg_data: data used to do double machine learning
    return: data frame containing all estimates for a given data set
    """
    le = preprocessing.LabelEncoder()
    le.fit(np.array(list(set(reg_data['V5']))))
    reg_data['V5_int_coded'] = le.transform(reg_data['V5'])
    
    # 7. nuisance stage: outcome ~ controls (use weights in 2)
    model = lgb.LGBMRegressor(num_boost_round=30, random_state=1)
    Y_hat = cross_val_predict(model, reg_data[all_features_lgb_without_C], reg_data['Y'], 
                              fit_params={
                                  'categorical_feature': cat_features_lgb,
                              },)    

    # 8. nuisance stage: treatment ~ controls (use weights in 2)
    # treatment = Z
    # controls: right hand side of reg_fmla except for Z
    model = lgb.LGBMClassifier(num_boost_round=30, random_state=1)
    Z_hat = cross_val_predict(model, reg_data[all_features_lgb_without_C], reg_data['Z'], 
                              fit_params={
                                  'categorical_feature': cat_features_lgb,
                              },method='predict_proba')[:,1]
    
    # 9. target stage: residualized outcome ~ residualized treatment (used weights in 2)
    # (Y - Y_hat) ~ (Z - Z_hat), where the hats are predicted values from 7 and 8
    # https://www.statsmodels.org/devel/examples/notebooks/generated/wls.html
    Y_residual = reg_data['Y'] - Y_hat
    Z_residual = reg_data['Z'] - Z_hat

    confidence_level = 0.90
    res_df = pd.DataFrame()

    # do overall satt
    small_df = pd.DataFrame()
    reg_fmla = 'Y_residual ~ -1 + Z_residual'
    res = smf.ols(formula = reg_fmla, data=reg_data).fit(cov_type='cluster',
            cov_kwds={'groups': reg_data['id.practice']})
    small_df['variable'] = pd.Series('Overall')
    small_df['level'] = pd.Series('NA')
    small_df['year'] = pd.Series('NA')
    small_df['satt'] = pd.Series(res.params['Z_residual'])
    small_df['lower90'] = pd.Series(res.params['Z_residual'] - norm.ppf(1 - (1 - confidence_level)/2)*res.bse['Z_residual'])
    small_df['upper90'] = pd.Series(res.params['Z_residual'] + norm.ppf(1 - (1 - confidence_level)/2)*res.bse['Z_residual'])

    res_df = res_df.append(small_df)

    # do by year satt
    small_df_3 = pd.DataFrame()
    small_df_4 = pd.DataFrame()
    reg_fmla = 'Y_residual ~ C(year):Z_residual'
    res = smf.ols(formula = reg_fmla, data=reg_data).fit(cov_type='cluster',
            cov_kwds={'groups': reg_data['id.practice']})

    small_df_3['variable'] = pd.Series('Overall')
    small_df_3['level'] = pd.Series('NA')
    small_df_3['year'] = pd.Series(3)
    small_df_3['satt'] = pd.Series(res.params['C(year)[3]:Z_residual'])
    small_df_3['lower90'] = pd.Series(res.params['C(year)[3]:Z_residual'] - norm.ppf(1 - (1 - confidence_level)/2)*res.bse['C(year)[3]:Z_residual'])
    small_df_3['upper90'] = pd.Series(res.params['C(year)[3]:Z_residual'] + norm.ppf(1 - (1 - confidence_level)/2)*res.bse['C(year)[3]:Z_residual'])

    small_df_4['variable'] = pd.Series('Overall')
    small_df_4['level'] = pd.Series('NA')
    small_df_4['year'] = pd.Series(4)
    small_df_4['satt'] = pd.Series(res.params['C(year)[4]:Z_residual'])
    small_df_4['lower90'] = pd.Series(res.params['C(year)[4]:Z_residual'] - norm.ppf(1 - (1 - confidence_level)/2)*res.bse['C(year)[4]:Z_residual'])
    small_df_4['upper90'] = pd.Series(res.params['C(year)[4]:Z_residual'] + norm.ppf(1 - (1 - confidence_level)/2)*res.bse['C(year)[4]:Z_residual'])

    res_df = res_df.append(small_df_3)
    res_df = res_df.append(small_df_4)

    # do group satt
    # X1
    small_df_X1_0 = pd.DataFrame()
    small_df_X1_1 = pd.DataFrame()

    reg_fmla = 'Y_residual ~ C(X1):Z_residual'
    res = smf.ols(formula = reg_fmla, data=reg_data).fit(cov_type='cluster',
            cov_kwds={'groups': reg_data['id.practice']})

    small_df_X1_0['variable'] = pd.Series('X1')
    small_df_X1_0['level'] = pd.Series('0')
    small_df_X1_0['year'] = pd.Series('NA')
    small_df_X1_0['satt'] = pd.Series(res.params['C(X1)[0]:Z_residual'])
    small_df_X1_0['lower90'] = pd.Series(res.params['C(X1)[0]:Z_residual'] - norm.ppf(1 - (1 - confidence_level)/2)*res.bse['C(X1)[0]:Z_residual'])
    small_df_X1_0['upper90'] = pd.Series(res.params['C(X1)[0]:Z_residual'] + norm.ppf(1 - (1 - confidence_level)/2)*res.bse['C(X1)[0]:Z_residual'])

    small_df_X1_1['variable'] = pd.Series('X1')
    small_df_X1_1['level'] = pd.Series('1')
    small_df_X1_1['year'] = pd.Series('NA')
    small_df_X1_1['satt'] = pd.Series(res.params['C(X1)[1]:Z_residual'])
    small_df_X1_1['lower90'] = pd.Series(res.params['C(X1)[1]:Z_residual'] - norm.ppf(1 - (1 - confidence_level)/2)*res.bse['C(X1)[1]:Z_residual'])
    small_df_X1_1['upper90'] = pd.Series(res.params['C(X1)[1]:Z_residual'] + norm.ppf(1 - (1 - confidence_level)/2)*res.bse['C(X1)[1]:Z_residual'])

    res_df = res_df.append(small_df_X1_0)
    res_df = res_df.append(small_df_X1_1)

    # do group satt
    # X2
    small_df_X2_A = pd.DataFrame()
    small_df_X2_B = pd.DataFrame()
    small_df_X2_C = pd.DataFrame()

    reg_fmla = 'Y_residual ~ C(X2):Z_residual'
    res = smf.ols(formula = reg_fmla, data=reg_data).fit(cov_type='cluster',
            cov_kwds={'groups': reg_data['id.practice']})

    small_df_X2_A['variable'] = pd.Series('X2')
    small_df_X2_A['level'] = pd.Series('A')
    small_df_X2_A['year'] = pd.Series('NA')
    small_df_X2_A['satt'] = pd.Series(res.params['C(X2)[A]:Z_residual'])
    small_df_X2_A['lower90'] = pd.Series(res.params['C(X2)[A]:Z_residual'] - norm.ppf(1 - (1 - confidence_level)/2)*res.bse['C(X2)[A]:Z_residual'])
    small_df_X2_A['upper90'] = pd.Series(res.params['C(X2)[A]:Z_residual'] + norm.ppf(1 - (1 - confidence_level)/2)*res.bse['C(X2)[A]:Z_residual'])

    small_df_X2_B['variable'] = pd.Series('X2')
    small_df_X2_B['level'] = pd.Series('B')
    small_df_X2_B['year'] = pd.Series('NA')
    small_df_X2_B['satt'] = pd.Series(res.params['C(X2)[B]:Z_residual'])
    small_df_X2_B['lower90'] = pd.Series(res.params['C(X2)[B]:Z_residual'] - norm.ppf(1 - (1 - confidence_level)/2)*res.bse['C(X2)[B]:Z_residual'])
    small_df_X2_B['upper90'] = pd.Series(res.params['C(X2)[B]:Z_residual'] + norm.ppf(1 - (1 - confidence_level)/2)*res.bse['C(X2)[B]:Z_residual'])

    small_df_X2_C['variable'] = pd.Series('X2')
    small_df_X2_C['level'] = pd.Series('C')
    small_df_X2_C['year'] = pd.Series('NA')
    small_df_X2_C['satt'] = pd.Series(res.params['C(X2)[C]:Z_residual'])
    small_df_X2_C['lower90'] = pd.Series(res.params['C(X2)[C]:Z_residual'] - norm.ppf(1 - (1 - confidence_level)/2)*res.bse['C(X2)[C]:Z_residual'])
    small_df_X2_C['upper90'] = pd.Series(res.params['C(X2)[C]:Z_residual'] + norm.ppf(1 - (1 - confidence_level)/2)*res.bse['C(X2)[C]:Z_residual'])

    res_df = res_df.append(small_df_X2_A)
    res_df = res_df.append(small_df_X2_B)
    res_df = res_df.append(small_df_X2_C)

    # do group satt
    # X3
    small_df_X3_0 = pd.DataFrame()
    small_df_X3_1 = pd.DataFrame()

    reg_fmla = 'Y_residual ~ C(X3):Z_residual'
    res = smf.ols(formula = reg_fmla, data=reg_data).fit(cov_type='cluster',
            cov_kwds={'groups': reg_data['id.practice']})

    small_df_X3_0['variable'] = pd.Series('X3')
    small_df_X3_0['level'] = pd.Series('0')
    small_df_X3_0['year'] = pd.Series('NA')
    small_df_X3_0['satt'] = pd.Series(res.params['C(X3)[0]:Z_residual'])
    small_df_X3_0['lower90'] = pd.Series(res.params['C(X3)[0]:Z_residual'] - norm.ppf(1 - (1 - confidence_level)/2)*res.bse['C(X3)[0]:Z_residual'])
    small_df_X3_0['upper90'] = pd.Series(res.params['C(X3)[0]:Z_residual'] + norm.ppf(1 - (1 - confidence_level)/2)*res.bse['C(X3)[0]:Z_residual'])

    small_df_X3_1['variable'] = pd.Series('X3')
    small_df_X3_1['level'] = pd.Series('1')
    small_df_X3_1['year'] = pd.Series('NA')
    small_df_X3_1['satt'] = pd.Series(res.params['C(X3)[1]:Z_residual'])
    small_df_X3_1['lower90'] = pd.Series(res.params['C(X3)[1]:Z_residual'] - norm.ppf(1 - (1 - confidence_level)/2)*res.bse['C(X3)[1]:Z_residual'])
    small_df_X3_1['upper90'] = pd.Series(res.params['C(X3)[1]:Z_residual'] + norm.ppf(1 - (1 - confidence_level)/2)*res.bse['C(X3)[1]:Z_residual'])

    res_df = res_df.append(small_df_X3_0)
    res_df = res_df.append(small_df_X3_1)

    # do group satt
    # X4
    small_df_X4_A = pd.DataFrame()
    small_df_X4_B = pd.DataFrame()
    small_df_X4_C = pd.DataFrame()

    reg_fmla = 'Y_residual ~ C(X4):Z_residual'
    res = smf.ols(formula = reg_fmla, data=reg_data).fit(cov_type='cluster',
            cov_kwds={'groups': reg_data['id.practice']})

    small_df_X4_A['variable'] = pd.Series('X4')
    small_df_X4_A['level'] = pd.Series('A')
    small_df_X4_A['year'] = pd.Series('NA')
    small_df_X4_A['satt'] = pd.Series(res.params['C(X4)[A]:Z_residual'])
    small_df_X4_A['lower90'] = pd.Series(res.params['C(X4)[A]:Z_residual'] - norm.ppf(1 - (1 - confidence_level)/2)*res.bse['C(X4)[A]:Z_residual'])
    small_df_X4_A['upper90'] = pd.Series(res.params['C(X4)[A]:Z_residual'] + norm.ppf(1 - (1 - confidence_level)/2)*res.bse['C(X4)[A]:Z_residual'])

    small_df_X4_B['variable'] = pd.Series('X4')
    small_df_X4_B['level'] = pd.Series('B')
    small_df_X4_B['year'] = pd.Series('NA')
    small_df_X4_B['satt'] = pd.Series(res.params['C(X4)[B]:Z_residual'])
    small_df_X4_B['lower90'] = pd.Series(res.params['C(X4)[B]:Z_residual'] - norm.ppf(1 - (1 - confidence_level)/2)*res.bse['C(X4)[B]:Z_residual'])
    small_df_X4_B['upper90'] = pd.Series(res.params['C(X4)[B]:Z_residual'] + norm.ppf(1 - (1 - confidence_level)/2)*res.bse['C(X4)[B]:Z_residual'])

    small_df_X4_C['variable'] = pd.Series('X4')
    small_df_X4_C['level'] = pd.Series('C')
    small_df_X4_C['year'] = pd.Series('NA')
    small_df_X4_C['satt'] = pd.Series(res.params['C(X4)[C]:Z_residual'])
    small_df_X4_C['lower90'] = pd.Series(res.params['C(X4)[C]:Z_residual'] - norm.ppf(1 - (1 - confidence_level)/2)*res.bse['C(X4)[C]:Z_residual'])
    small_df_X4_C['upper90'] = pd.Series(res.params['C(X4)[C]:Z_residual'] + norm.ppf(1 - (1 - confidence_level)/2)*res.bse['C(X4)[C]:Z_residual'])

    res_df = res_df.append(small_df_X4_A)
    res_df = res_df.append(small_df_X4_B)
    res_df = res_df.append(small_df_X4_C)

    # do group satt
    # X5
    small_df_X5_0 = pd.DataFrame()
    small_df_X5_1 = pd.DataFrame()

    reg_fmla = 'Y_residual ~ C(X5):Z_residual'
    res = smf.ols(formula = reg_fmla, data=reg_data).fit(cov_type='cluster',
            cov_kwds={'groups': reg_data['id.practice']})

    small_df_X5_0['variable'] = pd.Series('X5')
    small_df_X5_0['level'] = pd.Series('0')
    small_df_X5_0['year'] = pd.Series('NA')
    small_df_X5_0['satt'] = pd.Series(res.params['C(X5)[0]:Z_residual'])
    small_df_X5_0['lower90'] = pd.Series(res.params['C(X5)[0]:Z_residual'] - norm.ppf(1 - (1 - confidence_level)/2)*res.bse['C(X5)[0]:Z_residual'])
    small_df_X5_0['upper90'] = pd.Series(res.params['C(X5)[0]:Z_residual'] + norm.ppf(1 - (1 - confidence_level)/2)*res.bse['C(X5)[0]:Z_residual'])

    small_df_X5_1['variable'] = pd.Series('X5')
    small_df_X5_1['level'] = pd.Series('1')
    small_df_X5_1['year'] = pd.Series('NA')
    small_df_X5_1['satt'] = pd.Series(res.params['C(X5)[1]:Z_residual'])
    small_df_X5_1['lower90'] = pd.Series(res.params['C(X5)[1]:Z_residual'] - norm.ppf(1 - (1 - confidence_level)/2)*res.bse['C(X5)[1]:Z_residual'])
    small_df_X5_1['upper90'] = pd.Series(res.params['C(X5)[1]:Z_residual'] + norm.ppf(1 - (1 - confidence_level)/2)*res.bse['C(X5)[1]:Z_residual'])

    res_df = res_df.append(small_df_X5_0)
    res_df = res_df.append(small_df_X5_1)
    return res_df

def do_wls_full_pipeline(regression_data):
    """
    regression_data: data used to do wls
    return: data frame containing all estimates for a given data set
    """
    
    controls = ps_features_for_reg + V_features + pre_spending_features + time_features
    control_vars = ''
    for feature in controls:
        control_vars = control_vars + " + " + feature

    reg_data_drop_na = regression_data.dropna().reset_index(drop=True)

    confidence_level = 0.90
    res_df = pd.DataFrame()

    # do overall satt
    small_df = pd.DataFrame()
    reg_fmla = 'Y ~ Z' + control_vars
    res = smf.wls(formula = reg_fmla, data=reg_data_drop_na, weights=reg_data_drop_na['w']).fit(cov_type='cluster',
            cov_kwds={'groups': reg_data_drop_na['id.practice']})
    small_df['variable'] = pd.Series('Overall')
    small_df['level'] = pd.Series('NA')
    small_df['year'] = pd.Series('NA')
    small_df['satt'] = pd.Series(res.params['Z'])
    small_df['lower90'] = pd.Series(res.params['Z'] - norm.ppf(1 - (1 - confidence_level)/2)*res.bse['Z'])
    small_df['upper90'] = pd.Series(res.params['Z'] + norm.ppf(1 - (1 - confidence_level)/2)*res.bse['Z'])

    res_df = res_df.append(small_df)
    
    # do by year satt
    small_df_3 = pd.DataFrame()
    small_df_4 = pd.DataFrame()
    reg_fmla = 'Y ~ C(year):Z' + control_vars
    res = smf.wls(formula = reg_fmla, data=reg_data_drop_na, weights=reg_data_drop_na['w']).fit(cov_type='cluster',
            cov_kwds={'groups': reg_data_drop_na['id.practice']})

    small_df_3['variable'] = pd.Series('Overall')
    small_df_3['level'] = pd.Series('NA')
    small_df_3['year'] = pd.Series(3)
    small_df_3['satt'] = pd.Series(res.params['C(year)[3]:Z'])
    small_df_3['lower90'] = pd.Series(res.params['C(year)[3]:Z'] - norm.ppf(1 - (1 - confidence_level)/2)*res.bse['C(year)[3]:Z'])
    small_df_3['upper90'] = pd.Series(res.params['C(year)[3]:Z'] + norm.ppf(1 - (1 - confidence_level)/2)*res.bse['C(year)[3]:Z'])

    small_df_4['variable'] = pd.Series('Overall')
    small_df_4['level'] = pd.Series('NA')
    small_df_4['year'] = pd.Series(4)
    small_df_4['satt'] = pd.Series(res.params['C(year)[4]:Z'])
    small_df_4['lower90'] = pd.Series(res.params['C(year)[4]:Z'] - norm.ppf(1 - (1 - confidence_level)/2)*res.bse['C(year)[4]:Z'])
    small_df_4['upper90'] = pd.Series(res.params['C(year)[4]:Z'] + norm.ppf(1 - (1 - confidence_level)/2)*res.bse['C(year)[4]:Z'])

    res_df = res_df.append(small_df_3)
    res_df = res_df.append(small_df_4)
    
    # do group satt
    # X1
    small_df_X1_0 = pd.DataFrame()
    small_df_X1_1 = pd.DataFrame()

    reg_fmla = 'Y ~ C(X1):Z' + control_vars
    res = smf.wls(formula = reg_fmla, data=reg_data_drop_na, weights=reg_data_drop_na['w']).fit(cov_type='cluster',
            cov_kwds={'groups': reg_data_drop_na['id.practice']})

    small_df_X1_0['variable'] = pd.Series('X1')
    small_df_X1_0['level'] = pd.Series('0')
    small_df_X1_0['year'] = pd.Series('NA')
    small_df_X1_0['satt'] = pd.Series(res.params['C(X1)[0]:Z'])
    small_df_X1_0['lower90'] = pd.Series(res.params['C(X1)[0]:Z'] - norm.ppf(1 - (1 - confidence_level)/2)*res.bse['C(X1)[0]:Z'])
    small_df_X1_0['upper90'] = pd.Series(res.params['C(X1)[0]:Z'] + norm.ppf(1 - (1 - confidence_level)/2)*res.bse['C(X1)[0]:Z'])

    small_df_X1_1['variable'] = pd.Series('X1')
    small_df_X1_1['level'] = pd.Series('1')
    small_df_X1_1['year'] = pd.Series('NA')
    small_df_X1_1['satt'] = pd.Series(res.params['C(X1)[1]:Z'])
    small_df_X1_1['lower90'] = pd.Series(res.params['C(X1)[1]:Z'] - norm.ppf(1 - (1 - confidence_level)/2)*res.bse['C(X1)[1]:Z'])
    small_df_X1_1['upper90'] = pd.Series(res.params['C(X1)[1]:Z'] + norm.ppf(1 - (1 - confidence_level)/2)*res.bse['C(X1)[1]:Z'])

    res_df = res_df.append(small_df_X1_0)
    res_df = res_df.append(small_df_X1_1)

    # do group satt
    # X2
    small_df_X2_A = pd.DataFrame()
    small_df_X2_B = pd.DataFrame()
    small_df_X2_C = pd.DataFrame()

    reg_fmla = 'Y ~ C(X2):Z' + control_vars
    res = smf.wls(formula = reg_fmla, data=reg_data_drop_na, weights=reg_data_drop_na['w']).fit(cov_type='cluster',
            cov_kwds={'groups': reg_data_drop_na['id.practice']})

    small_df_X2_A['variable'] = pd.Series('X2')
    small_df_X2_A['level'] = pd.Series('A')
    small_df_X2_A['year'] = pd.Series('NA')
    small_df_X2_A['satt'] = pd.Series(res.params['C(X2)[A]:Z'])
    small_df_X2_A['lower90'] = pd.Series(res.params['C(X2)[A]:Z'] - norm.ppf(1 - (1 - confidence_level)/2)*res.bse['C(X2)[A]:Z'])
    small_df_X2_A['upper90'] = pd.Series(res.params['C(X2)[A]:Z'] + norm.ppf(1 - (1 - confidence_level)/2)*res.bse['C(X2)[A]:Z'])

    small_df_X2_B['variable'] = pd.Series('X2')
    small_df_X2_B['level'] = pd.Series('B')
    small_df_X2_B['year'] = pd.Series('NA')
    small_df_X2_B['satt'] = pd.Series(res.params['C(X2)[B]:Z'])
    small_df_X2_B['lower90'] = pd.Series(res.params['C(X2)[B]:Z'] - norm.ppf(1 - (1 - confidence_level)/2)*res.bse['C(X2)[B]:Z'])
    small_df_X2_B['upper90'] = pd.Series(res.params['C(X2)[B]:Z'] + norm.ppf(1 - (1 - confidence_level)/2)*res.bse['C(X2)[B]:Z'])

    small_df_X2_C['variable'] = pd.Series('X2')
    small_df_X2_C['level'] = pd.Series('C')
    small_df_X2_C['year'] = pd.Series('NA')
    small_df_X2_C['satt'] = pd.Series(res.params['C(X2)[C]:Z'])
    small_df_X2_C['lower90'] = pd.Series(res.params['C(X2)[C]:Z'] - norm.ppf(1 - (1 - confidence_level)/2)*res.bse['C(X2)[C]:Z'])
    small_df_X2_C['upper90'] = pd.Series(res.params['C(X2)[C]:Z'] + norm.ppf(1 - (1 - confidence_level)/2)*res.bse['C(X2)[C]:Z'])

    res_df = res_df.append(small_df_X2_A)
    res_df = res_df.append(small_df_X2_B)
    res_df = res_df.append(small_df_X2_C)

    # do group satt
    # X3
    small_df_X3_0 = pd.DataFrame()
    small_df_X3_1 = pd.DataFrame()

    reg_fmla = 'Y ~ C(X3):Z' + control_vars
    res = smf.wls(formula = reg_fmla, data=reg_data_drop_na, weights=reg_data_drop_na['w']).fit(cov_type='cluster',
            cov_kwds={'groups': reg_data_drop_na['id.practice']})

    small_df_X3_0['variable'] = pd.Series('X3')
    small_df_X3_0['level'] = pd.Series('0')
    small_df_X3_0['year'] = pd.Series('NA')
    small_df_X3_0['satt'] = pd.Series(res.params['C(X3)[0]:Z'])
    small_df_X3_0['lower90'] = pd.Series(res.params['C(X3)[0]:Z'] - norm.ppf(1 - (1 - confidence_level)/2)*res.bse['C(X3)[0]:Z'])
    small_df_X3_0['upper90'] = pd.Series(res.params['C(X3)[0]:Z'] + norm.ppf(1 - (1 - confidence_level)/2)*res.bse['C(X3)[0]:Z'])

    small_df_X3_1['variable'] = pd.Series('X3')
    small_df_X3_1['level'] = pd.Series('1')
    small_df_X3_1['year'] = pd.Series('NA')
    small_df_X3_1['satt'] = pd.Series(res.params['C(X3)[1]:Z'])
    small_df_X3_1['lower90'] = pd.Series(res.params['C(X3)[1]:Z'] - norm.ppf(1 - (1 - confidence_level)/2)*res.bse['C(X3)[1]:Z'])
    small_df_X3_1['upper90'] = pd.Series(res.params['C(X3)[1]:Z'] + norm.ppf(1 - (1 - confidence_level)/2)*res.bse['C(X3)[1]:Z'])

    res_df = res_df.append(small_df_X3_0)
    res_df = res_df.append(small_df_X3_1)

    # do group satt
    # X4
    small_df_X4_A = pd.DataFrame()
    small_df_X4_B = pd.DataFrame()
    small_df_X4_C = pd.DataFrame()

    reg_fmla = 'Y ~ C(X4):Z' + control_vars
    res = smf.wls(formula = reg_fmla, data=reg_data_drop_na, weights=reg_data_drop_na['w']).fit(cov_type='cluster',
            cov_kwds={'groups': reg_data_drop_na['id.practice']})

    small_df_X4_A['variable'] = pd.Series('X4')
    small_df_X4_A['level'] = pd.Series('A')
    small_df_X4_A['year'] = pd.Series('NA')
    small_df_X4_A['satt'] = pd.Series(res.params['C(X4)[A]:Z'])
    small_df_X4_A['lower90'] = pd.Series(res.params['C(X4)[A]:Z'] - norm.ppf(1 - (1 - confidence_level)/2)*res.bse['C(X4)[A]:Z'])
    small_df_X4_A['upper90'] = pd.Series(res.params['C(X4)[A]:Z'] + norm.ppf(1 - (1 - confidence_level)/2)*res.bse['C(X4)[A]:Z'])

    small_df_X4_B['variable'] = pd.Series('X4')
    small_df_X4_B['level'] = pd.Series('B')
    small_df_X4_B['year'] = pd.Series('NA')
    small_df_X4_B['satt'] = pd.Series(res.params['C(X4)[B]:Z'])
    small_df_X4_B['lower90'] = pd.Series(res.params['C(X4)[B]:Z'] - norm.ppf(1 - (1 - confidence_level)/2)*res.bse['C(X4)[B]:Z'])
    small_df_X4_B['upper90'] = pd.Series(res.params['C(X4)[B]:Z'] + norm.ppf(1 - (1 - confidence_level)/2)*res.bse['C(X4)[B]:Z'])

    small_df_X4_C['variable'] = pd.Series('X4')
    small_df_X4_C['level'] = pd.Series('C')
    small_df_X4_C['year'] = pd.Series('NA')
    small_df_X4_C['satt'] = pd.Series(res.params['C(X4)[C]:Z'])
    small_df_X4_C['lower90'] = pd.Series(res.params['C(X4)[C]:Z'] - norm.ppf(1 - (1 - confidence_level)/2)*res.bse['C(X4)[C]:Z'])
    small_df_X4_C['upper90'] = pd.Series(res.params['C(X4)[C]:Z'] + norm.ppf(1 - (1 - confidence_level)/2)*res.bse['C(X4)[C]:Z'])

    res_df = res_df.append(small_df_X4_A)
    res_df = res_df.append(small_df_X4_B)
    res_df = res_df.append(small_df_X4_C)

    # do group satt
    # X5
    small_df_X5_0 = pd.DataFrame()
    small_df_X5_1 = pd.DataFrame()

    reg_fmla = 'Y ~ C(X5):Z' + control_vars
    res = smf.wls(formula = reg_fmla, data=reg_data_drop_na, weights=reg_data_drop_na['w']).fit(cov_type='cluster',
            cov_kwds={'groups': reg_data_drop_na['id.practice']})

    small_df_X5_0['variable'] = pd.Series('X5')
    small_df_X5_0['level'] = pd.Series('0')
    small_df_X5_0['year'] = pd.Series('NA')
    small_df_X5_0['satt'] = pd.Series(res.params['C(X5)[0]:Z'])
    small_df_X5_0['lower90'] = pd.Series(res.params['C(X5)[0]:Z'] - norm.ppf(1 - (1 - confidence_level)/2)*res.bse['C(X5)[0]:Z'])
    small_df_X5_0['upper90'] = pd.Series(res.params['C(X5)[0]:Z'] + norm.ppf(1 - (1 - confidence_level)/2)*res.bse['C(X5)[0]:Z'])

    small_df_X5_1['variable'] = pd.Series('X5')
    small_df_X5_1['level'] = pd.Series('1')
    small_df_X5_1['year'] = pd.Series('NA')
    small_df_X5_1['satt'] = pd.Series(res.params['C(X5)[1]:Z'])
    small_df_X5_1['lower90'] = pd.Series(res.params['C(X5)[1]:Z'] - norm.ppf(1 - (1 - confidence_level)/2)*res.bse['C(X5)[1]:Z'])
    small_df_X5_1['upper90'] = pd.Series(res.params['C(X5)[1]:Z'] + norm.ppf(1 - (1 - confidence_level)/2)*res.bse['C(X5)[1]:Z'])

    res_df = res_df.append(small_df_X5_0)
    res_df = res_df.append(small_df_X5_1)    
    
    return res_df

