"""

    Classificação de um XOR de 3 dimensões

    Informações : iremos classificar dados
    onde iremos dizer se o dado possui a aresta
    de um cubo no plano R3

"""

from pybrain.datasets.supervised import SupervisedDataSet
from pybrain.tools.shortcuts import buildNetwork
from pybrain.supervised.trainers.backprop import BackpropTrainer

def montaDados ( ) :
    """

    Função na qual monta o dataset

    :return: dataset montado
    """
    dataset = SupervisedDataSet(3, 1)

    dataset.addSample ( [0, 0, 0], [0] )
    dataset.addSample ( [0, 1, 1], [0] )
    dataset.addSample ( [1, 0, 1], [0] )
    dataset.addSample ( [1, 1, 0], [0] )
    dataset.addSample ( [1, 0, 0], [1] )
    dataset.addSample ( [0, 0, 1], [1] )
    dataset.addSample ( [0, 1, 0], [0] )
    dataset.addSample ( [1, 1, 1], [1] )

    return dataset

def main (  ) :

    # criando os dados para treino
    datasetTreino = montaDados ()

    # criando os dados para teste
    datasetTeste = montaDados()

    # definindo a estrutura de como será a rede neural
    # a entrada será a dimensão de entrada do dataset = 3
    # terá 6 neurônios na primeira camada intermediária
    # terá 6 neurônios na segunda camada escondida
    # terá como dimensão de saída o tamanho do dado de saída = 1
    # terá a função de autocorreção para melhor adaptação da rede
    network = buildNetwork( datasetTreino.indim, 12, 6, datasetTreino.outdim, bias=True )

    # criando a rede neural
    # terá como estrutura de rede neural definida no objeto network
    # utilizará os dados do dataset para treino
    neuralNetwork = BackpropTrainer ( network, datasetTreino, learningrate=0.01, momentum=0.9 )

    # treinando a rede
    neuralNetwork.trainEpochs ( 1500 )

    # validando a rede
    neuralNetwork.testOnData ( datasetTeste, verbose=True )

if __name__ == "__main__" :

    main()

