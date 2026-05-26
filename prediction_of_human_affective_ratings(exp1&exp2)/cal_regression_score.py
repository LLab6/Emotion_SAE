# umap 

"""
Created on Mon May 19 11:32:28 2025

@author: xiuwenwu
"""
import numpy as np
import pickle
import numpy as np
import matplotlib.pyplot as plt
import os 
# from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
import argparse

def generate_regression_score(model,sae_layer,sae_width,language,M,
                        save_root_path):

    
    data_path = os.path.join('./',f'sae_features_{model}',f'{sae_layer}_{sae_width}',f'top_10_sae_dict_{language}.pkl')

    with open(data_path, 'rb') as f:
        data = pickle.load(f)

    sae_container = []
    labels = []

    for id, value in data.items():
        if id == 25:
            continue
        sae_container.extend(value)

        if id == 26:
            id -=1
        labels.extend([id] * value.shape[0])
    sae_container = np.array(sae_container)

    labels = np.array(labels)


    save_file_path = os.path.join(save_root_path,model,f'{sae_layer}_{sae_width}')

    os.makedirs(save_file_path, exist_ok=True)
    print(save_file_path)

    if M == 0:

        clf = LogisticRegression(multi_class='multinomial', max_iter=1000)
        acc_true = cross_val_score(clf, sae_container, labels, cv=5, scoring='accuracy').mean()

        value_container = [acc_true]
        save_value_path = os.path.join(save_file_path,f'value_{language}_all_space_regression.npy')
        np.save(save_value_path,value_container)



    np.random.seed(M)
    y_shuf = np.random.permutation(labels)
    clf = LogisticRegression(multi_class='multinomial', max_iter=1000)
    acc_perm= cross_val_score(clf, sae_container, y_shuf, cv=5, scoring='accuracy').mean()

    # 6. 
    print(data_path)


    save_perm_value_path = os.path.join(save_file_path,f'value_{language}_all_space_regression_acc_perm_{M}.npy')
    np.save(save_perm_value_path,acc_perm)

    # plt.figure()
    # plt.hist(acc_perm, bins=30)
    # plt.axvline(acc_true, linestyle='--')
    # plt.title('Regression Acc Null Distribution')
    # plt.xlabel('Regression Acc')
    # plt.ylabel('Frequency')
    # save_plt_value_path = os.path.join(save_file_path,f'value_{language}_all_space_regression_acc_plt.png')
    # plt.savefig(save_plt_value_path)
    # plt.clf()



def main(args):
    # parameters
    model = args.model
    sae_layer = args.sae_layer
    sae_width = args.sae_width
    language =  args.language
    output_path = args.output_path
    M = args.M

    generate_regression_score(model,sae_layer,sae_width,language,M,output_path)



if __name__ == '__main__':



    parser = argparse.ArgumentParser(
        prog='regression sae space', # 程序名
        description='parameters', # 描述
    )

    # model = ['gemma','llama']
    parser.add_argument('--model',default='llama')

    # sae_layer = [0.25,0.5.0.75]
    parser.add_argument('--sae_layer',default=0.25,type=float)

    # sae_width = [1,2]
    parser.add_argument('--sae_width',default=1,type=int)

    # sae_language=['ch','en']
    parser.add_argument('--language',default='ch',type=str)

    # M in range(0,1000)
    parser.add_argument('--M',default=0,type=int)

    # './save_path'
    parser.add_argument('--output_path',default='./save_path',type=str)


    args = parser.parse_args()

    main(args)


