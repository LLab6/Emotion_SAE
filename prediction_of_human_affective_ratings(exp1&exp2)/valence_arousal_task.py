import argparse
import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy.stats import spearmanr, pearsonr
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import os 

def min_max_normalize_1d(array):
 
    if len(array.shape) != 1:
        raise ValueError("Input data must be a 1D array.")
    

    array_min = array.min()
    array_max = array.max()
    
   
    array_range = array_max - array_min
    if array_range == 0:
        return np.zeros_like(array)  
    
    # maxmin norm
    normalized_array = (array - array_min) / array_range

    return normalized_array

def load_emo_data(task):

    # 1 -> valence c-c c-e
    # 2 -> valence e-e e-c
    # 3 -> arousal c-c c-e
    # 4 -> arousal e-e e-c


    data_path_e = "./EEMO.csv"

    df_e = pd.read_csv(data_path_e)

    words_e = df_e['Word'].to_list()

    valence_value_e = df_e['V.Mean.Sum'].to_numpy()

    arousal_value_e = df_e['A.Mean.Sum'].to_numpy()

    data_path_c = "./CEMO.csv"

    df_c = pd.read_csv(data_path_c)

    words_c = df_c['Word'].to_list()

    valence_value_c = df_c['Valence_Mean'].to_numpy()

    arousal_value_c = df_c['Arousal_Mean'].to_numpy()

    ch_en_path = "./CE_EMO_emotion.en.ch.csv"

    df = pd.read_csv(ch_en_path)
    dwords_e = df['word.EN'].to_list()
    dwords_c = df['word.CH'].to_list()

    if task == 1 or task == 3:

        word_dict = dict(zip(dwords_c, dwords_e))

        re_word_dict = dict(zip(dwords_e, dwords_c))

    elif task == 2 or task == 4:

        word_dict = dict(zip(dwords_e, dwords_c))

        re_word_dict = dict(zip(dwords_c, dwords_e))


    if task == 1 or task == 2:

        return words_e,valence_value_e,words_c,valence_value_c,word_dict,re_word_dict
    
    if task == 3 or task == 4 :

        return words_e,arousal_value_e,words_c,arousal_value_c,word_dict,re_word_dict
    

def load_sae_data(model,sae_layer,sae_width,sae_space):



    sae_c_file_path = os.path.join('./sae_data',f'sae_features_{model}',f'{sae_layer}_{sae_width}','sae_acts_c.npy')

    sae_e_file_path = os.path.join('./sae_data',f'sae_features_{model}',f'{sae_layer}_{sae_width}','sae_acts_e.npy')

    sae_c = np.load(sae_c_file_path)

    sae_e = np.load(sae_e_file_path)

    if sae_space == 'all':

        return sae_e,sae_c
    
    else:

        sae_space_path = os.path.join('./sae_data',f'sae_features_{model}',f'{sae_layer}_{sae_width}',f'{sae_space}_dimensions.npy')

        dimensions = np.load(sae_space_path)

        sae_e = sae_e[:,dimensions]

        sae_c = sae_c[:,dimensions]

        return sae_e,sae_c
        
def training_xgboost(words_e,value_e,words_c,value_c,sae_e,sae_c,
                     word_dict,re_word_dict,
                     model_name,sae_layer,sae_width,sae_space,task,seed,
                     output_path):

    # 1 -> valence ch-ch ch-eng
    # 2 -> valence eng-eng eng-ch
    # 3 -> arousal ch-ch ch-eng
    # 4 -> arousal eng-eng eng-ch

    if task == 1 or task == 3:
        # 中文是训练集
        X_sparse_train = sae_c  # 稀疏表示的特征矩阵
        
        y_value_train = value_c # 目标值
        y_value_train = min_max_normalize_1d(y_value_train)

        #英文作为测试集
        X_sparse_test = sae_e

        y_value_test = value_e
        y_value_test = min_max_normalize_1d(y_value_test)

        words_main = words_c
        words_another  = words_e

    elif task == 2 or task == 4:

        X_sparse_train = sae_e  # 稀疏表示的特征矩阵
        
        y_value_train = value_e # 目标值
        y_value_train = min_max_normalize_1d(y_value_train)

        #英文作为测试集
        X_sparse_test = sae_c

        y_value_test = value_c
        y_value_test = min_max_normalize_1d(y_value_test)

        words_main = words_e
        words_another  = words_c


        # config LightGBM parameters
    params = {
        "objective": "regression",        
        "metric": "rmse",                
        "learning_rate": 0.01,            
        "max_depth": -1,                  
        "num_leaves": 31,               
        "feature_fraction": 0.8,         
        "bagging_fraction": 0.8,         
        "bagging_freq": 5,                
        "lambda_l1": 0.0,                
        "lambda_l2": 0.0,              
        "verbose": -1,                 
        "seed": seed                      
    }

    # five fold validation
    kf = KFold(n_splits=5, shuffle=True, random_state=seed)


    importance_df_list = []  

    eval_cor_list = []

    test_cor_list = []

    eval_results_save = []
    test_results_save = []
    eval_ground_truth = []
    test_ground_truth = []

    for fold, (train_index, val_index) in enumerate(kf.split(X_sparse_train)):

        print(f"Fold {fold+1}")
        
        # 划分训练集和验证集
        X_train, X_val = X_sparse_train[train_index], X_sparse_train[val_index]
        y_train, y_val = y_value_train[train_index], y_value_train[val_index]

        val_word = [words_main[index] for index in val_index]

        val_another_language_words = [word_dict[word] for word in val_word if word in word_dict]


        test_indices = [words_another.index(word) for word in val_another_language_words if word in words_another]


        found_another_words = [word for word in val_another_language_words if word in words_another]

 
        found_main_words = [re_word_dict[word] for word in found_another_words]


        val_indices = [words_main.index(word) for word in found_main_words if word in words_main]



        # val
        X_val = X_sparse_train[val_indices]
        y_val = y_value_train[val_indices]

        # # init LightGBM dataset

        dtrain = lgb.Dataset(X_train, label=y_train, free_raw_data=False)
        dval = lgb.Dataset(X_val, label=y_val, free_raw_data=False)
        
        # training model
        model = lgb.train(params, dtrain, num_boost_round=2000, valid_sets=[dval])

        # eval
        y_pred = model.predict(X_val, num_iteration=model.best_iteration)
        rmse = np.sqrt(mean_squared_error(y_val, y_pred))
        r2 = r2_score(y_val, y_pred)
        
        # print output
        print('Chinese Words Performance:')
        print(f"Fold {fold+1} - RMSE: {rmse:.4f}, R2: {r2:.4f}")
        pearson_corr_val, pearson_p_val = pearsonr(y_val, y_pred)
        spearman_corr_val, spearman_p_val = spearmanr(y_val, y_pred)
        print(f"Pearson Correlation: {pearson_corr_val} (p-value: {pearson_p_val})")
        print(f"Spearman Correlation: {spearman_corr_val} (p-value: {spearman_p_val})")
        print('\n')

        # test set
        x_test = X_sparse_test[test_indices]
        y_test = y_value_test[test_indices]

        # y_pred_e = model.predict(X_sparse_e,num_iteration=model.best_iteration)
        y_pred_test = model.predict(x_test,num_iteration=model.best_iteration)


        rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
        r2_test = r2_score(y_test, y_pred_test)
        print('English Words Performance:')
        print(f"Fold {fold+1} - RMSE: {rmse_test:.4f}, R2: {r2_test:.4f}")
        pearson_corr_test, pearson_p_test = pearsonr(y_test, y_pred_test)
        spearman_corr_test, spearman_p_test = spearmanr(y_test, y_pred_test)
        print(f"Pearson Correlation: {pearson_corr_test} (p-value: {pearson_p_test})")
        print(f"Spearman Correlation: {spearman_corr_test} (p-value: {spearman_p_test})")
        print('\n')

        eval_cor_list.append(pearson_corr_val)
        test_cor_list.append(pearson_corr_test)


        # importance index
        importance_dict = model.feature_importance(importance_type='split')
        importance_df = pd.DataFrame({
            "Feature": [f"f{i}" for i in range(X_sparse_train.shape[1])],
            "Importance": importance_dict,
            "Fold": fold+1
        })
        importance_df_list.append(importance_df)    

        eval_results_save.extend(y_pred)
        test_results_save.extend(y_pred_test)
        eval_ground_truth.extend(y_val)
        test_ground_truth.extend(y_test)
    
    pred_data = {'y_pred': eval_results_save,'y_pred_test': test_results_save,'y_val': eval_ground_truth,'y_test': test_ground_truth}
    pred_df = pd.DataFrame(pred_data)

    save_file_path = os.path.join(output_path,f'{model_name}_sae_layer_{sae_layer}_sae_width_{sae_width}_importance_files_{sae_space}')
    if not os.path.exists(save_file_path):
        os.makedirs(save_file_path)

    pred_df.to_csv(os.path.join(save_file_path,f"task_{task}_pred_results_seed_{seed}.csv"), index=False)


    all_importances_df = pd.concat(importance_df_list, ignore_index=True)


    all_importances_df.to_csv(os.path.join(save_file_path,f"task_{task}_all_importance_results_seed_{seed}.csv"), index=False)

    eval_cor_list = np.array(eval_cor_list)
    test_cor_list = np.array(test_cor_list)
    eval_cor = np.mean(eval_cor_list)
    test_cor = np.mean(test_cor_list)

    np.save(os.path.join(save_file_path,f'task_{task}_seed_{seed}_cor.npy'),[eval_cor,test_cor])

    return eval_cor,test_cor


def main(args):

    # parameters
    model = args.model
    sae_layer = args.sae_layer
    sae_width = args.sae_width
    task =  args.task
    seed = args.seed
    sae_space = args.sae_space
    output_path = args.output_path

    # load data

    words_e, value_e, words_c, value_c, word_dict, re_word_dict = load_emo_data(task)

    sae_e,sae_c = load_sae_data(model,sae_layer,sae_width,sae_space)

    eval_cor,tests_cor = training_xgboost(words_e,value_e,words_c,value_c,sae_e,sae_c,
                     word_dict,re_word_dict,
                     model,sae_layer,sae_width,sae_space,task,seed,
                     output_path)
    


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        prog='valence_arousal_prediction', # 程序名
        description='parameters', # 描述
    )

    # model = ['gemma','llama']
    parser.add_argument('--model',default='llama')

    # sae_layer = [0.25,0.5.0.75]
    parser.add_argument('--sae_layer',default=0.25,type=float)

    # sae_width = [1,2]
    parser.add_argument('--sae_width',default=1,type=int)

    # sae_space=['all','common','none_common']
    parser.add_argument('--sae_space',default='all',type=str)

    # task = [1,2,3,4]
    # 1 -> valence c-c c-e
    # 2 -> valence e-e e-c
    # 3 -> arousal c-c c-e
    # 4 -> arousal e-e e-c

    parser.add_argument('--task',default=1,type=int)

    # seed = [3, 6, 44, 53, 57, 69, 77, 78, 90, 94]
    parser.add_argument('--seed',default=3,type=int)

    # './save_path'
    parser.add_argument('--output_path',default='./save_path',type=str)


    args = parser.parse_args()

    main(args)
