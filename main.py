from kmodes.kmodes import KModes
from Preprocessing import *
from K_Elbow import *
import KModes_Model as KModes_Model
import pickle
import pandas as pd
import Model

if __name__=="__main__":
    
    # Load the model
    with open('kmodes_model.pkl', 'rb') as file:
        model = pickle.load(file)

    path = r"D:\B3\Group Project\USTHConnect\dataset.csv"
    df = pd.read_csv(path)
    
    assign_cluster_df = Model.assign_cluster(model, df)
    Model.recommend_cluster(model, df.iloc[0], assign_cluster_df)