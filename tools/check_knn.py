from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

from characterdataset.oshifinder.predict import KNN_classifier



character_folder = "data\character_embedds"
knn = KNN_classifier(character_folder, n_neighbors=4)

#t-SNEで次元削減
tsne = TSNE(n_components=2, random_state = 0, perplexity = 30, n_iter = 1000)
if len(knn.embeddings.shape) > 2:
    embeddings_cls = knn.embeddings.squeeze(1)

normed_embedding = tsne.fit_transform(knn.embeddings)
print(normed_embedding.shape)


fig = plt.figure(figsize=(10, 10))

colors = {f"{next(iter(set(knn.labels)))}": "red", f"{list(set(knn.labels))[-1]}": "blue"}
print(colors)
for i in range(normed_embedding.shape[0]):
    plt.scatter(normed_embedding[i, 0], normed_embedding[i, 1],
                c=colors[knn.labels[i]])

plt.xlim((normed_embedding[:, 0].min(), normed_embedding[:, 1].max()))
plt.ylim((normed_embedding[:, 1].min(), normed_embedding[:, 1].max()))

plt.show()
