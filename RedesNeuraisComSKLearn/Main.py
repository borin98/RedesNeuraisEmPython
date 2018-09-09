"""

    Implementação da rede neural
    utilizando o sklearn

    Onde iremos utilizar o dataset do
    iris como exemplo

    Informação : A classificação dos dados de saída serão da forma :

        Iris-setosa = 0
        Iris-versicolor = 1
        Iris-virginica = 2


"""

from sklearn import datasets
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split

def main (  ) :

    # criando o dataset
    iris = datasets.load_iris()

    # separando os dados de entrada e saida
    dadosEntrada, dadosSaida = iris.data, iris.target

    # criando a estrutura da rede neural
    # utilizando o algorítmo adam para otimização dos pesos
    # parâmetro que ajuda a evitar o overfiting
    # 5 neurônios na camada escondida
    networkMLP = MLPClassifier ( solver="adam", alpha=0.0001, hidden_layer_sizes = ( 5, ), random_state=1,
                                 learning_rate="constant", learning_rate_init=0.01, max_iter=100,
                                 activation="logistic", momentum=0.9, verbose=True, tol=0.001 )

    # separando os dados para treino e teste
    # onde iremos separar 70 % dos dados para treino
    # e 30 % dos dados para teste
    xTreino, xTeste, yTreino, yTeste = train_test_split(
        dadosEntrada, dadosSaida, test_size=0.3, random_state=1
    )

    networkMLP.fit ( xTreino, yTreino )

    # fazendo a classificação dos dados
    saidas = networkMLP.predict ( xTeste )

    print(10*""+"\n")
    print("Saída da rede : {}\t".format ( saidas ) )
    print("Saída esperada : {}\t".format ( yTeste ) )
    print("\nScore da rede : {} % de acerto".format ( 100 * ( networkMLP.score ( xTeste, yTeste ) ) ) )

if __name__ == '__main__':
    main()