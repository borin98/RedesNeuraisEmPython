"""

"""
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix

def main (  ) :

    # carregando o dataset
    dataset = load_breast_cancer()

    # definindo a escala do objeto
    escala = StandardScaler()

    # criando a estrutura da rede neural
    # terão 3 camadas escondidas, com 3 neurônios cada uma
    neuralNetworlMLP = MLPClassifier ( hidden_layer_sizes= ( 30, 30, 30 ),max_iter=1000 )

    # criando os dados de entrada e saida
    entradaDados, saidaDados = dataset["data"], dataset["target"]

    # fazendo o split dos dados de treino e teste
    # onde 70 % dos dados serão para treino
    # e 20 % dos dados serão para teste
    xTreino, xTeste, yTreino, yTeste = train_test_split(
        entradaDados, saidaDados, train_size=0.3
    )

    # ajustando a escala
    escala.fit ( xTreino )

    # atribuindo os valores normalizados aos datasets de entrada
    xTreino = escala.transform ( xTreino )
    xTeste = escala.transform ( xTeste )

    # ajustando a rede neural
    neuralNetworlMLP.fit ( xTreino, yTreino )

    # fazendo o dataset das predições da rede
    predicao = neuralNetworlMLP.predict ( xTeste )

    print("Precisão da máquina : {} % de acertos".format ( 100*neuralNetworlMLP.score(
        xTeste, yTeste
    ) ) )

    # fazendo a verificação do modelo
    print( classification_report ( yTeste, predicao ) )

    # extraindo os pesos da rede neural
    # na forma de um array
    vetor = neuralNetworlMLP.coefs_
    print(vetor)

if __name__ == '__main__':
    main()