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
cachorro3 = [1, 1, 0]

dados = [porco1, porco2, porco3, cachorro1, cachorro2, cachorro3]
# 1 = porco, 0 = cachorro
classes = [1, 1, 1, 0, 0, 0] 

model = LinearSVC()
model.fit(dados, classes)

# Features(Características)
# pelo longo (1 - sim, 0 não)
# perna curta (1 - sim, 0 não)
# faz auau (1 - sim, 0 não)
misterio1 = [1, 1, 1]
misterio2 = [0, 0, 0]
misterio3 = [0, 1, 0]
misterio4 = [1, 0, 0]
misterios = [misterio1, misterio2, misterio3, misterio4]
misterios_classes = [0, 1, 1, 0]

previsoes = model.predict(misterios)
taxa_acertos = accuracy_score(misterios_classes, previsoes)
print(taxa_acertos)