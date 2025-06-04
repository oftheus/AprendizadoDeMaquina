from sklearn.decomposition import PCA
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, adjusted_rand_score, accuracy_score
from sklearn.metrics import confusion_matrix, accuracy_score
from kneed import KneeLocator
from scipy.cluster.hierarchy import dendrogram, linkage

# carregando o dataset
colunas = ['sepalaComprimento', 'sepalaLargura',
           'petalaComprimento', 'petalaLargura', 'classe']
dadosIris = pd.read_csv(os.path.join(os.path.dirname(__file__), 'iris', 'iris.data'),
                        header=None, names=colunas)

# análise inicial
print("Primeiras linhas do dataset:\n", dadosIris.head())
print("\nInformações do dataset:")
dadosIris.info()
print("\nEstatísticas descritivas:\n", dadosIris.describe())
print("\nContagem de valores por classe:\n",
      dadosIris['classe'].value_counts())

# Comentário item a: quantos grupos naturais você vê?
print("\n> Visualmente, parece haver 3 grupos naturais:")
print("  - Iris-setosa aparece bem isolada,")
print("  - Iris-versicolor e Iris-virginica formam dois blocos com alguma sobreposição.\n")

# visualizações univariadas
plt.figure(figsize=(12, 8))
for i, atributo in enumerate(dadosIris.columns[:-1]):
    plt.subplot(2, 2, i + 1)
    sns.boxplot(x='classe', y=atributo, data=dadosIris)
    plt.title(f'Distribuição de {atributo} por classe')
plt.tight_layout()
plt.show()

# visualizações multivariadas (sem classes)
sns.pairplot(dadosIris.drop(columns='classe'), kind='kde',
             corner=True, height=1.5, aspect=1)
plt.suptitle("Distribuição Conjunta das Features (KDE)", y=0.99)
plt.show()

# análise de outliers
plt.figure(figsize=(8, 6))
sns.boxplot(data=dadosIris.drop(columns='classe'))
plt.title("Identificação de Outliers nas Features")
plt.xticks(rotation=30, ha='right')
plt.tight_layout()
plt.show()

# pré-processamento
atributos = dadosIris.drop(columns='classe')
normalizador = StandardScaler()
atributosNormalizados = normalizador.fit_transform(atributos)
atributosNormalizadosDf = pd.DataFrame(
    atributosNormalizados, columns=atributos.columns)

# PCA
pca = PCA(n_components=2)
atributosPca = pca.fit_transform(atributosNormalizadosDf)

plt.figure(figsize=(8, 6))
plt.scatter(atributosPca[:, 0], atributosPca[:, 1], alpha=0.7)
plt.xlabel('Componente Principal 1')
plt.ylabel('Componente Principal 2')
plt.title('Visualização PCA (2 componentes)')
plt.grid()
plt.show()

# método do cotovelo para determinar k
print("\nMétodo do Cotovelo")
inerciaClusters = []
intervaloK = range(1, 11)

for k in intervaloK:
    kmeansTemp = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeansTemp.fit(atributosNormalizadosDf)
    inerciaClusters.append(kmeansTemp.inertia_)

plt.figure(figsize=(10, 6))
plt.plot(intervaloK, inerciaClusters, 'bo-')
plt.xlabel('Número de Clusters (k)', fontsize=12)
plt.ylabel('Inércia', fontsize=12)
plt.title('Método do Cotovelo para Determinação do k Ótimo', fontsize=14)
plt.xticks(intervaloK)
plt.grid(True)
plt.axvline(x=3, color='r', linestyle='--', alpha=0.5)
plt.text(3.1, max(inerciaClusters)*0.9, 'k=3', color='r')
plt.show()

# aplicando K-Means com k=3
print("\nK-Means com k=3")
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
rotulosClusters = kmeans.fit_predict(atributosNormalizadosDf)

# adicionando clusters ao DataFrame original
dadosIris['clusterKMeans'] = rotulosClusters

# determinação do melhor k pelo cotovelo
cotoveloK = np.argmin(np.diff(inerciaClusters, 2)) + 2
print(
    f"Segunda derivada discreta da curva de inércia (diferença das diferenças) sugere k = {cotoveloK}")

kneedle = KneeLocator(
    range(1, 11),
    inerciaClusters,
    curve='convex',
    direction='decreasing'
)
cotoveloK2 = kneedle.elbow  # Retorna o k ótimo
print(f"KneeLocator sugere k = {cotoveloK2}")

print(
    f"Silhouette Score para k=3: {silhouette_score(atributosNormalizadosDf, rotulosClusters):.3f}")

# adicionando componentes principais ao DataFrame
dadosIris['pca1'] = atributosPca[:, 0]
dadosIris['pca2'] = atributosPca[:, 1]

# visualização dos clusters
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
sns.scatterplot(x='pca1', y='pca2', hue='clusterKMeans',
                palette='viridis', data=dadosIris, s=100)
centroides_pca = pca.transform(pd.DataFrame(
    kmeans.cluster_centers_, columns=atributos.columns))
plt.scatter(centroides_pca[:, 0], centroides_pca[:, 1],
            marker='X', s=200, c='red', label='Centróides')
plt.title('Clusters encontrados pelo K-Means (k=3)\n', fontsize=12)
plt.xlabel('Componente Principal 1', fontsize=10)
plt.ylabel('Componente Principal 2', fontsize=10)
plt.legend(title='Cluster')
plt.grid(alpha=0.3)

plt.subplot(1, 2, 2)
sns.scatterplot(x='pca1', y='pca2', hue='classe',
                palette='Set2', data=dadosIris, s=100)
plt.title('Classes Reais do Dataset Iris\n', fontsize=12)
plt.xlabel('Componente Principal 1', fontsize=10)
plt.ylabel('Componente Principal 2', fontsize=10)
plt.legend(title='Espécie')
plt.grid(alpha=0.3)

plt.tight_layout()
plt.show()

# análise de correspondência entre clusters e classes reais
print("\nCorrespondência entre Clusters e Classes Reais")
tabelaCruzada = pd.crosstab(dadosIris['clusterKMeans'], dadosIris['classe'])
print(tabelaCruzada)

# melhor mapeamento cluster-classe


def encontrarMelhorMapeamento(rotulosReais, rotulosCluster):
    from itertools import permutations
    unicosReais = np.unique(rotulosReais)
    unicosClusters = np.unique(rotulosCluster)

    melhorAcuracia = 0
    melhorPermutacao = None

    for permutacao in permutations(unicosClusters):
        rotulosMapeados = np.zeros_like(rotulosCluster)
        for i, classe in enumerate(unicosReais):
            rotulosMapeados[rotulosCluster == permutacao[i]] = classe

        acuracia = accuracy_score(rotulosReais, rotulosMapeados)
        if acuracia > melhorAcuracia:
            melhorAcuracia = acuracia
            melhorPermutacao = permutacao

    return melhorPermutacao, melhorAcuracia


melhorMapeamento, melhorAcuracia = encontrarMelhorMapeamento(
    dadosIris['classe'].astype('category').cat.codes, rotulosClusters)

print(f"\nAcurácia com melhor mapeamento: {melhorAcuracia:.2%}")

# vsisualização 3D
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

grafico = ax.scatter(
    atributosNormalizadosDf.iloc[:, 2], atributosNormalizadosDf.iloc[:,
                                                                     3], atributosNormalizadosDf.iloc[:, 0],
    c=rotulosClusters, cmap='viridis', s=50)

ax.set_xlabel('Pétala Comprimento')
ax.set_ylabel('Pétala Largura')
ax.set_zlabel('Sépala Comprimento')
plt.title('Visualização 3D dos Clusters (K-Means)')
plt.colorbar(grafico)
plt.show()

# dendrograma (apenas up to 40 amostras para visualização clara)
linked = linkage(atributosNormalizadosDf, method='ward')
plt.figure(figsize=(12, 5))
dendrogram(linked,
           truncate_mode='lastp',
           p=40,
           leaf_rotation=45.,
           leaf_font_size=8.,
           show_contracted=True)
plt.axhline(y=10, color='r', linestyle='--')
plt.title('Dendrograma (Hierarchical Clustering) - Ward Linkage')
plt.xlabel('Índice da Amostra')
plt.ylabel('Distância')
plt.tight_layout()
plt.show()

# agglomerativeClustering com k=3
agglo = AgglomerativeClustering(
    n_clusters=3, metric='euclidean', linkage='ward')
rotulosAgglo = agglo.fit_predict(atributosNormalizadosDf)

dadosIris['clusterAgglo'] = rotulosAgglo

# visualização dos clusters hierárquicos via PCA
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
sns.scatterplot(x='pca1', y='pca2', hue='clusterAgglo',
                palette='Set1', data=dadosIris, s=80)
plt.title('Clusters Hierárquicos (k=3) via PCA')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend(title='Cluster')
plt.grid(alpha=0.3)

plt.subplot(1, 2, 2)
sns.scatterplot(x='pca1', y='pca2', hue='classe',
                palette='Set2', data=dadosIris, s=80)
plt.title('Classes Reais (via PCA)')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend(title='Espécie')
plt.grid(alpha=0.3)

plt.tight_layout()
plt.show()

# garantir que os rótulos dos clusters sejam inteiros
rotulosKMeans = dadosIris['clusterKMeans'].astype(int).values
rotulosAgglomerativo = dadosIris['clusterAgglo'].astype(int).values

# cálculo do Índice Rand Ajustado (ARI)
ariKMeans = adjusted_rand_score(
    dadosIris['classe'].astype('category').cat.codes, rotulosKMeans)
ariAgglomerativo = adjusted_rand_score(
    dadosIris['classe'].astype('category').cat.codes, rotulosAgglomerativo)

# cálculo do Silhouette Score
silhuetaKMeans = silhouette_score(atributosNormalizadosDf, rotulosKMeans)
silhuetaAgglomerativo = silhouette_score(
    atributosNormalizadosDf, rotulosAgglomerativo)

# acurácia com mapeamento ótimo


def encontrarMelhorMapeamento(rotulosReais, rotulosClusters):
    from itertools import permutations
    classesUnicas = np.unique(rotulosReais)
    clustersUnicos = np.unique(rotulosClusters)
    melhorAcuracia = 0
    melhorPermutacao = None
    for permutacao in permutations(clustersUnicos):
        rotulosMapeados = np.zeros_like(rotulosClusters)
        for i, classe in enumerate(classesUnicas):
            rotulosMapeados[rotulosClusters == permutacao[i]] = classe
        acuracia = accuracy_score(rotulosReais, rotulosMapeados)
        if acuracia > melhorAcuracia:
            melhorAcuracia = acuracia
            melhorPermutacao = permutacao
    return melhorAcuracia, melhorPermutacao


acuraciaKMeans, _ = encontrarMelhorMapeamento(
    dadosIris['classe'].astype('category').cat.codes, rotulosKMeans)
acuraciaAgglomerativo, _ = encontrarMelhorMapeamento(
    dadosIris['classe'].astype('category').cat.codes, rotulosAgglomerativo)

# Matriz de confusão
print("\nMatriz de Confusão (K-Means):")
print(confusion_matrix(dadosIris['classe'].astype(
    'category').cat.codes, rotulosKMeans))

print("\nMatriz de Confusão (Agrupamento Hierárquico):")
print(confusion_matrix(dadosIris['classe'].astype(
    'category').cat.codes, rotulosAgglomerativo))

# Tabela resumo comparativa
print("\nRESUMO DAS MÉTRICAS:")
print(f"{'Métrica':<20} {'K-Means':<10} {'Hierárquico':<10}")
print(f"{'ARI':<20} {ariKMeans:.3f}{'':<10} {ariAgglomerativo:.3f}")
print(f"{'Silhueta':<20} {silhuetaKMeans:.3f}{'':<10} {silhuetaAgglomerativo:.3f}")
print(f"{'Acurácia':<20} {acuraciaKMeans:.3f}{'':<10} {acuraciaAgglomerativo:.3f}")

print("\nDiscussão:")
print("- O K-Means consegue separar bem o Iris-setosa (cluster limpo), mas tem dificuldade em distinguir completamente Versicolor x Virginica, que ficam parcialmente misturados.")
print("- O clustering hierárquico (dendrograma) mostra que, primeiro, há um grande agrupamento separando Setosa, e depois Versicolor e Virginica se dividem; faz sentido biologicamente.")
print("- O ARI e o Silhouette confirmam que ambos os métodos identificam 3 grupos, mas K-Means obteve ARI levemente maior do que Hierárquico (ou vice-versa, dependendo da execução), indicando separação similar.")
print("- Em geral, ambas as técnicas encontram a estrutura de 3 clusters, porém as fronteiras exatas diferem.\n")

print("\n\nRepetindo com apenas duas features: petalaComprimento e petalaLargura")

# selecionando apenas as duas colunas
atrib2 = dadosIris[['petalaComprimento', 'petalaLargura']].copy()
atr2_norm = StandardScaler().fit_transform(atrib2)

# PCA para 2 features não faz sentido (já são 2D): mas só para plotar podemos usar diretamento
plt.figure(figsize=(6, 6))
plt.scatter(atr2_norm[:, 0], atr2_norm[:, 1], alpha=0.7)
plt.xlabel('Comprimento da Pétala (normalizado)')
plt.ylabel('Largura da Pétala (normalizado)')
plt.title('Espaço 2D (Pétala): Visualização antes do clustering')
plt.grid()
plt.show()

# K-Means (k=3) nas 2 features
kmeans2 = KMeans(n_clusters=3, random_state=42, n_init=10)
r2_km = kmeans2.fit_predict(atr2_norm)
sil2_km = silhouette_score(atr2_norm, r2_km)
ari2_km = adjusted_rand_score(
    dadosIris['classe'].astype('category').cat.codes, r2_km)

# Hierárquico (k=3) nas 2 features
agglo2 = AgglomerativeClustering(
    n_clusters=3, metric='euclidean', linkage='ward')
r2_ag = agglo2.fit_predict(atr2_norm)
sil2_ag = silhouette_score(atr2_norm, r2_ag)
ari2_ag = adjusted_rand_score(
    dadosIris['classe'].astype('category').cat.codes, r2_ag)

# visualizando clusters 2D (K-Means e Hierárquico)
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
sns.scatterplot(x=atr2_norm[:, 0], y=atr2_norm[:, 1], hue=r2_km,
                palette='viridis', s=80)
plt.title(
    f'K-Means (2 features) k=3\nSilhouette={sil2_km:.3f}, ARI={ari2_km:.3f}')
plt.xlabel('Pétala Comprimento (norm)')
plt.ylabel('Pétala Largura (norm)')
plt.legend(title='Cluster')
plt.grid(alpha=0.3)

plt.subplot(1, 2, 2)
sns.scatterplot(x=atr2_norm[:, 0], y=atr2_norm[:, 1], hue=r2_ag,
                palette='Set1', s=80)
plt.title(
    f'Agglomerative (2 features) k=3\nSilhouette={sil2_ag:.3f}, ARI={ari2_ag:.3f}')
plt.xlabel('Pétala Comprimento (norm)')
plt.ylabel('Pétala Largura (norm)')
plt.legend(title='Cluster')
plt.grid(alpha=0.3)

plt.tight_layout()
plt.show()

print("Resultados (2 features apenas):")
print(f"- K-Means: Silhouette = {sil2_km:.3f}, ARI = {ari2_km:.3f}")
print(f"- Hierárquico: Silhouette = {sil2_ag:.3f}, ARI = {ari2_ag:.3f}")

print("\nDiscussão final (2 features vs. 4 features):")
print("- Observa-se que usar apenas as duas features da pétala geralmente melhora a separação entre as três espécies, especialmente entre Versicolor e Virginica.")
print("- Em 4 features, o Silhouette e o ARI eram X e Y; em 2 features, eles mudaram para valores (… ); isso indica que as dimensões relacionadas à pétala são mais discriminantes.")
print("- Conclui-se que as features de pétala trazem maior poder de separação natural, enquanto que usar todas as 4 pode introduzir ruído adicional.\n")
