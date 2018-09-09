"""

    Iremos fazer uma rede neural, na qual irá utilizar
    o dataset da iris para fazer a classificação dos dados
    utilizando uma rede neural

    Informação dos dados :

        A classificação é da forma :

        Iris-setosa = 1
        Iris-versicolor = 2
        Iris-virginica = 3

"""
import numpy as np
from pybrain.datasets.supervised import SupervisedDataSet
from pybrain.tools.shortcuts import buildNetwork
from pybrain.supervised.trainers.backprop import BackpropTrainer

def treinaRede ( entradaTreino, saidaTreino ) :
    """
    Função que cria o método de treino da rede

    :param entradaTreino: dados de entrada do treino
    :param saidaTreino: dados de saída do treino
    :return: treinamento : objeto que diz qual será o treino da rede
    """
    # serão 4 dados de entrada
    # será um dado de saída
    treinamento = SupervisedDataSet(4, 1)

    for i in range(len(entradaTreino)):

        treinamento.addSample(entradaTreino[i], saidaTreino[i])

    return treinamento

def montaRede ( dadosEntrada, dadosSaida ) :
    """
    Função na qual def

    :param dadosEntrada: parâmetros de entrada na rede neural
    :param dadosSaida:  parâmetros de saída da rede neural
    :return: retorna a rede de treinamento treinada e os dados supervisionados
    """

    entradaTreino = np.concatenate ( ( dadosEntrada[:35], dadosEntrada[50:85],dadosEntrada[100:135] ) )
    saidaTreino = np.concatenate ( ( dadosSaida[:35], dadosSaida[50:85], dadosSaida[100:135] ) )
    entradaTeste = np.concatenate ( ( dadosEntrada[35:50], dadosEntrada[85:100], dadosEntrada[135:] ) )
    saidaTeste = np.concatenate ( ( dadosSaida[35:50], dadosSaida[85:100], dadosSaida[135:] ) )

    treinaRede ( entradaTreino, saidaTreino )

    # criando o dataset de treinamento
    # serão 4 dados de entrada
    # será um dado de saída
    treinamento = treinaRede ( entradaTreino, saidaTreino )

    # rede neural do tamanho do treinamento
    # com 2 neurônios na camada intermediária
    # com o dado de output sendo o tamanho da rede
    # utilizando bias
    redeNeural = buildNetwork ( treinamento.indim, 2, treinamento.outdim, bias=True )

    # criando a rede neural treinada
    redeNeuralTreinada = BackpropTrainer ( redeNeural, treinamento, learningrate=0.3, momentum=0.9)

    for epocas in range ( 0, 10000 ) :

        redeNeuralTreinada.train()

    teste = SupervisedDataSet ( 4, 1 )

    for i in range ( len ( entradaTeste ) ) :

        teste.addSample ( entradaTeste[i], saidaTeste[i] )

    return redeNeuralTreinada, teste


def main (  ) :

    # carregando os dados utilizando o numpy
    dadosEntrada = np.genfromtxt("Iris.data", delimiter=",", usecols= ( 0, 1, 2, 3 ) )
    dadosSaida = np.genfromtxt ( "Iris.data", delimiter=",", usecols= ( 4 ) )

    redeNeural, dadosTeste = montaRede ( dadosEntrada, dadosSaida )

    # realizando o teste
    redeNeural.testOnData ( dadosTeste, verbose=True )

if ( __name__=="__main__" ) :

    main()

