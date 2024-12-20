from kmodes.kmodes import KModes
from sklearn.metrics import silhouette_score, davies_bouldin_score
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import joblib
from Preprocessing import *

# Some problems occur in this function, currently just save and load model
def training_model(df, n_clusters, init='Huang', n_init=10, random_state=42):
    train_data = data_preprocess(df)
    km = KModes(n_clusters=n_clusters, init=init, n_init=n_init, random_state=random_state)
    km.fit(train_data)
    clusters = km.predict(train_data)
    
    df["Cluster"] = clusters
    
    # Save the K-Modes model
    model_filename = "kmodes_model.pkl"
    joblib.dump(km, model_filename)
    print(f"Model saved as {model_filename}")
    
    return km, df
    
def recommend_cluster(model, data, df):
    """_summary_

    Args:
        model
        data: data of student to recommend (unprocessed)
        df: dataframe after processing with cluster column
    """
    train_data = transfer_data(data)
    pred = model.predict(train_data)
    
    print(f"Recommendation for {data.FullName}:")
    stu_idx = 1
    for idx, row in df[df["Cluster"] == pred[0]].iterrows():
        print(f"{stu_idx}. {row['FullName']}")
        stu_idx += 1