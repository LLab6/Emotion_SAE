import os 
import json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr

def get_fold_cor(sub_df,seed,seed_length_df):

    start_idx = 0
    chance_level_pearson_fold = []
    chance_level_pearson_fold_test = []

    for index_length in seed_length_df[str(seed)].to_list():

        y_pred_fold = sub_df.iloc[start_idx:start_idx+index_length]['y_pred']
        # y_pred_c = sub_df['y_pred_c']

        y_val_fold = sub_df.iloc[start_idx:start_idx+index_length]['y_val']


        y_test_fold = sub_df.iloc[start_idx:start_idx+index_length]['y_test']
        if 'y_pred_test' in sub_df.keys():
            y_pred_re_fold = sub_df.iloc[start_idx:start_idx+index_length]['y_pred_test']



        n_iterations = 10000
        shuffled_y_vals = []
        chance_level_pearson_min = []
        for _ in range(n_iterations):
            shuffled = y_val_fold.sample(frac=1).reset_index(drop=True)
            shuffled_y_vals.append(shuffled)
            shuffled_value = shuffled.to_list()
            pearson_corr_e, pearson_p_e = pearsonr(shuffled_value, y_pred_fold.to_list())
            
            chance_level_pearson_min.append(pearson_corr_e)

        chance_level_pearson_fold.append(chance_level_pearson_min)

        shuffled_y_test = []
        chance_level_pearson_min_test = []
        for _ in range(n_iterations):
            shuffled_test = y_test_fold.sample(frac=1).reset_index(drop=True)
            shuffled_y_test.append(shuffled_test)
            shuffled_value_test = shuffled_test.to_list()
            pearson_corr_e_test, pearson_p_e_test = pearsonr(shuffled_value_test, y_pred_re_fold.to_list())

            chance_level_pearson_min_test.append(pearson_corr_e_test)

        chance_level_pearson_fold_test.append(chance_level_pearson_min_test)

        start_idx += index_length
    
    return chance_level_pearson_fold,chance_level_pearson_fold_test

if __name__ == '__main__':
    
    output_path = './save_path'
    model_names = ['llama','gemma']
    sae_layers = [0.25,0.50,0.75]
    sae_width = [1,2]
    sae_space = ['common','none_common','all']
    tasks = [1,2,3,4]


    for model_name in model_names:

        for sae_layer in sae_layers:

            for sae_w in sae_width:

                for sae_s in sae_space:
        
                    file_path = os.path.join(output_path,f'{model_name}_sae_layer_{sae_layer}_sae_width_{sae_w}_importance_files_{sae_s}')
                    print(file_path)
                    files = os.listdir(file_path)

                    for task in tasks:

                        seed_length_path = "./seed_length.csv"
                        seed_length_df = pd.read_csv(seed_length_path)
                        seeds = [3, 6, 44, 53, 57, 69, 77, 78, 90, 94]
                        chance_level_pearson_seed = []
                        chance_level_pearson_seed_test = []

                        for seed in seeds:
                            f_path = os.path.join(file_path,f"task_{task}_pred_results_seed_{seed}.csv")
                            df = pd.read_csv(f_path)
                            # print(df.iloc[200:300]['y_pred'].to_list())
                            results,results_test = get_fold_cor(df,seed,seed_length_df)
                            results = np.array(results)
                            results = np.mean(results,axis=0)

                            results_test = np.array(results_test)
                            results_test = np.mean(results_test,axis=0)
            
                            chance_level_pearson_seed.append(results)

                            chance_level_pearson_seed_test.append(results_test)


                        chance_level_pearson_seed = np.array(chance_level_pearson_seed)
                        chance_level_pearson_seed = np.mean(chance_level_pearson_seed,axis=0)


                        chance_level_pearson_seed_test = np.array(chance_level_pearson_seed_test)
                        chance_level_pearson_seed_test = np.mean(chance_level_pearson_seed_test,axis=0)

                        save_dict = {'chance_level':chance_level_pearson_seed,
                                    'chance_level_test':chance_level_pearson_seed_test}
                        save_df = pd.DataFrame(save_dict)
                        save_df.to_csv(os.path.join(file_path,f'task_{task}_chance_level.csv'),index=False)