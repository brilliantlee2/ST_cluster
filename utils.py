import os
from graph_func import graph_construction
from utils_func import  load_ST_file
from args_parser import set_parser
from collections import defaultdict
params = set_parser()
import pandas as pd
import numpy as np

def load_data(data_id):

	data_root = './spatial_datasets/DLPFC/'
	data_test = load_ST_file(file_fold=os.path.join(data_root, str(data_id)))
	graph_dict_tmp = graph_construction(data_test.obsm['spatial'], data_test.shape[0], params)
	df_label = pd.read_csv(os.path.join(data_root, str(data_id), "metadata.tsv"), sep='\t')
	data_test.obs['layer_guess'] = np.array(df_label['layer_guess'].to_list())

	return data_test

def load_graph(data):
	graph_dict = graph_construction(data.obsm['spatial'], data.shape[0], params)
	
	return graph_dict


def convert_str_to_int(ad_sc, annotation = "layer_guess"):
# The labelled cell type has to be in the adata.obs
    type_label_dict = defaultdict()
    num = 0
    labels = []
    for i, j in enumerate(ad_sc.obs[annotation]):
        if j not in type_label_dict.keys():
            type_label_dict[j] = num
            labels.append(num)
            num += 1
        else:
            labels.append(type_label_dict[j])

    return labels, type_label_dict