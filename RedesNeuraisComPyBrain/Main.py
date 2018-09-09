"""
    Implementando uma rede de múltiplas camadas usando a
    biblioteca do pybrain

    Onde esta rede é uma camada de Feed Forward
    Modelo muito utilizado em redes neurais MLP

"""

from pybrain.datasets.supervised import SupervisedDataSet
from pybrain.tools.shortcuts import buildNetwork
from pybrain.supervised.trainers.backprop import BackpropTrainer

def criandoDataset ( dataset ) :
    """

    Função que adiciona as amostras para a rede

    :return: void
    """
    # criando os exemplos
    dataset.addSample ( [1, 1], [0] )
    dataset.addSample ( [0, 1], [1] )
    dataset.addSample ( [1, 0], [1] )
    dataset.addSample ( [0, 0], [0] )

def main (  ) :

    # criando o dataset, onde os dados de entrada no dataset será um vetor de tamanho 2
    # e o dado de saída será um escalar
    dataset = SupervisedDataSet ( 2, 1 )

    criandoDataset ( dataset )

    # criando a rede neural
    # onde terá, respectivamente, a quantidade de entrada na rede
    # quantidade de neurônios na camada intermediária
    # dimensão de saída da rede
    # utilizando uma adaptação da rede ao longo do tempo
    network = buildNetwork ( dataset.indim, 4, dataset.outdim, bias=True)

    # criando o método de treino da rede
    # passando a rede
    # passando o dataset
    # passando a taxa de aprendizado
    # aumentando o cálculo que maximiza o treinamento da rede
    trainer = BackpropTrainer ( network, dataset, learningrate=0.01, momentum=0.99 )

    # looping que faz o treino da  função
    for epocas in range ( 0, 1000 ) :

        trainer.train()

    # realizando o teste
    datasetTeste = SupervisedDataSet ( 2, 1 )
    criandoDataset ( datasetTeste )
    trainer.testOnData ( datasetTeste, verbose=True )


if ( __name__ == "__main__") :

    main()