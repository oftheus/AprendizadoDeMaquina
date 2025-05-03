import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# definindo o nome das colunas
colunas = ['sepalaComprimento', 'sepalaLargura',
           'petalaComprimento', 'petalaLargura', 'classe']

# carregando o dataset iris
irisDataSet = pd.read_csv(os.path.join(os.path.dirname(
    __file__), 'iris', 'iris.data'), header=None, names=colunas)


'''
exibindo as 10 primeiras linhas do dataset
para verificar como os dados estão organizados
e ver se os nomes das colunas estão certos
'''
print(irisDataSet.head(10))

# análise exploratória
print("\nEstatísticas descritivas:")
print(irisDataSet.describe())

# verificando como as classes estão distribuídas
quantidadePorClasse = irisDataSet['classe'].value_counts()
print("\nDistribuição das Classes no Dataset Iris:\n", quantidadePorClasse)
# plotando um gráfico pizza para uma melhor vizualização
plt.figure(figsize=(4, 4))
plt.pie(quantidadePorClasse, labels=quantidadePorClasse,
        autopct='%1.1f%%', startangle=100)
plt.title('Distribuição das Classes no Dataset Iris')
plt.axis('equal')

# visualizando as classes do dataset de maneira completa com o pairplot
sns.pairplot(irisDataSet, hue='classe')

# dividindo os dados em dados de treinamento e teste
# aqui removemos a coluna classe para usar só os atributos como entrada
x = irisDataSet.drop('classe', axis=1)
y = irisDataSet['classe']  # definimos a coluna classe como variável alvo
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.3, random_state=42)  # 70% pro treino e 30% pro teste

# criando o modelo de árvore de decisão
modelo = DecisionTreeClassifier(criterion='entropy', random_state=42)

# treinando o modelo
modelo.fit(x_train, y_train)

# fazendo as previsões
yPrevisto = modelo.predict(x_test)

# avaliando desempenho do modelo
# percentual de acertos totais
print("\nAcurácia:", accuracy_score(y_test, yPrevisto))
# mostrando onde o modelo arcertou e onde errou, matriz de confusão
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix(y_test, yPrevisto),
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=modelo.classes_,
            yticklabels=modelo.classes_)
plt.title('Matriz de Confusão')
plt.xlabel('Previsto')
plt.ylabel('Real')

# mostrando as métricas, precisão, recall, f1-score...
print("\nRelatório de classificação:", classification_report(y_test, yPrevisto))

# visualizando a árvore
plt.figure(figsize=(16, 8))
plot_tree(modelo, feature_names=x.columns,
          class_names=modelo.classes_, filled=True)
plt.title("Árvore de Decisão Treinada - Dataset Iris")
plt.show()

# realizando 10-fold Cross Validation
scores = cross_val_score(modelo, x, y, cv=10)
print("Acurácia média (10-fold):", scores.mean())
print("Desvio padrão:", scores.std())

# testando com diferentes hiperparâmetros
print("\nTestando com diferentes hiperparâmetros")
for criterio in ['gini', 'entropy']:
    for profundidadeMaxima in [1, 2, 3]:
        modeloTeste = DecisionTreeClassifier(
            criterion=criterio,
            max_depth=profundidadeMaxima,
            random_state=42
        )
        modeloTeste.fit(x_train, y_train)
        yPrevistoTeste = modeloTeste.predict(x_test)
        acuracia = accuracy_score(y_test, yPrevistoTeste)
        print(
            f"Critério: {criterio}, Profundidade Máxima: {profundidadeMaxima}, Acurácia: {acuracia:.3f}")
