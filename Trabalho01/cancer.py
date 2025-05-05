import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix

# definindo o nome das colunas
colunas = [
    'id', 'diagnostico',
    'raio_medio', 'textura_media', 'perimetro_medio', 'area_media', 'suavidade_media',
    'compacidade_media', 'concavidade_media', 'pontos_concavos_medios', 'simetria_media', 'dimensao_fractal_media',
    'raio_se', 'textura_se', 'perimetro_se', 'area_se', 'suavidade_se',
    'compacidade_se', 'concavidade_se', 'pontos_concavos_se', 'simetria_se', 'dimensao_fractal_se',
    'raio_pior', 'textura_pior', 'perimetro_pior', 'area_pior', 'suavidade_pior',
    'compacidade_pior', 'concavidade_pior', 'pontos_concavos_pior', 'simetria_pior', 'dimensao_fractal_pior'
]

# carregando o dataset breast cancer
cancerDataSet = pd.read_csv(os.path.join(os.path.dirname(
    __file__), 'breastCancer', 'wdbc.data'), header=None, names=colunas)

'''
exibindo as 10 primeiras linhas do dataset
para verificar como os dados estão organizados
e ver se os nomes das colunas estão certos
'''
print(cancerDataSet.head(10))

# análise exploratória
print("\nEstatísticas descritivas:")
print(cancerDataSet.describe())

# Gráfico de pizza
cancerDataSet['diagnostico'].value_counts().plot.pie(
    labels=["Benigno", "Maligno"], autopct="%.1f%%", colors=["blue", "yellow"])
plt.title("Distribuição das Classes")
plt.show()

# Pairplot
sns.pairplot(cancerDataSet.iloc[:, [0, 1, 2, 3, -1]],
             hue='diagnostico', palette='Set1')
plt.show()

# Preparar dados
x = cancerDataSet.drop('diagnostico', axis=1)
y = cancerDataSet['diagnostico']

# Dividir treino e teste
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.3, random_state=42)

# Testar diferentes profundidades e critérios
criterios = ['gini', 'entropy']
for criterio in criterios:
    for max_depth in [1, 2, 3]:
        clf = DecisionTreeClassifier(criterion=criterio, max_depth=max_depth)
        clf.fit(x_train, y_train)
        acc = clf.score(x_test, y_test)
        print(
            f"Critério: {criterio}, Profundidade: {max_depth}, Acurácia: {acc:.3f}")

# Melhor modelo (exemplo)
clf = DecisionTreeClassifier(criterion='entropy', max_depth=3)
clf.fit(x_train, y_train)
y_pred = clf.predict(x_test)

# Avaliação
print(classification_report(y_test, y_pred))
sns.heatmap(confusion_matrix(y_test, y_pred),
            annot=True, fmt="d", cmap="Blues")
plt.xlabel("Previsto")
plt.ylabel("Real")
plt.title("Matriz de Confusão")
plt.show()

# Validação cruzada
scores = cross_val_score(clf, x, y, cv=10)
print(f"Validação cruzada (10-fold): Média de acurácia = {scores.mean():.3f}")
