from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

# Features(Características)
# pelo longo (1 - sim, 0 não)
# perna curta (1 - sim, 0 não)
# faz auau (1 - sim, 0 não)
porco1 = [0, 1, 0]
porco2 = [0, 1, 1]
porco3 = [1, 1, 0]
cachorro1 = [0, 1, 1]
cachorro2 = [1, 0, 1]
cachorro3 = [1, 1, 1]

treino_x = [porco1, porco2, porco3, cachorro1, cachorro2, cachorro3]
treino_y = [1, 1, 1, 0, 0, 0]# 1 = porco, 0 = cachorro

model = LinearSVC()
model.fit(treino_x, treino_y)

misterio1 = [1, 1, 1]
misterio2 = [0, 0, 0]
misterio3 = [0, 1, 0]
misterio4 = [1, 0, 0]

teste_x = [misterio1, misterio2, misterio3, misterio4]
teste_y = [0, 1, 1, 0]
previsoes = model.predict(teste_x)

taxa_acertos = accuracy_score(teste_y, previsoes)
print(taxa_acertos)