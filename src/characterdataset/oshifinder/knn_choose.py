from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from characterdataset.configs import load_global_config


def run():
    config = load_global_config()

    audio_embds_dir = os.path.join(config.finder.character_embedds, "embeddings")
    embeddings = None
    labels = []
    dim = 0
    # これはサブフォルダの名前をリストに格納する
    role_dirs = []
    for item in os.listdir(audio_embds_dir):
        if os.path.isdir(os.path.join(audio_embds_dir, item)):
            role_dirs.append(item)


    for role_dir in role_dirs:
        role = os.path.basename(os.path.normpath(role_dir))
        
        file_list = [os.path.join(audio_embds_dir, role_dir, embeddings_path) for embeddings_path in os.listdir(os.path.join(audio_embds_dir, role_dir))]
        
        for embeddings_path in file_list:
            with open(embeddings_path, 'rb') as fp:
                embedding = pickle.load(fp)
            fp.close()
            
            # 前作ったリストに格納する
            if dim == 0:
                embeddings_cls = embedding
                dim = embeddings_cls.shape[0]
            else:
                # This is equivalent to concatenation along the first axis after 1-D arrays of shape (N,) have been reshaped to (1,N)
                embeddings_cls = np.vstack((embeddings_cls, embedding))
            
            labels.append(role)

    print(embeddings_cls.shape, len(labels))


    # 探索するクラスタ数の範囲を設定
    range_clusters = range(2, 10)

    best_score = 0
    best_k = 2
    scores = []

    for k in range_clusters:
        kmeans = KMeans(n_clusters=k, n_init="auto", random_state=0)
        labels = kmeans.fit_predict(embeddings_cls.squeeze(1))
        

        # シルエットスコアを計算
        score = silhouette_score(embeddings_cls.squeeze(1), labels)
        scores.append(score)
        
        # ベストスコアの更新
        if score > best_score:
            best_score = score
            best_k = k

    print("Best k:", best_k)
    print("Best silhouette score:", best_score)


    # シルエットスコアのプロット
    plt.figure(figsize=(10,6))
    plt.plot(range_clusters, scores, marker='o')
    plt.xlabel('Number of clusters')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Score for Different Number of Clusters')
    plt.show()

if __name__ == "__main__":
    run()