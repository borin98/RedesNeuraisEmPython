"""

    Fazendo a implementação de uma rede múltipla camada
    onde iremos classificar os dados do iris.dataset

    Informação : A classificação dos dados de saída serão da forma :

        Iris-setosa = 0
        Iris-versicolor = 1
        Iris-virginica = 2

"""
from sklearn import datasets
from pybrain.datasets.classification import ClassificationDataSet
from pybrain.tools.shortcuts import buildNetwork
from pybrain.supervised.trainers.backprop import BackpropTrainer
import matplotlib.pyplot as plt
from pybrain.structure.modules.softmax import SoftmaxLayer
from pybrain.utilities import percentError

def montaDatasetConvertido ( dadosTemporario ) :
    """
    função que converte o objeto
    python.datasets.classficication.ClassificationDataSet
    para python.datasets.supervised.SupervisedDataSet

    Será utilizando tanto para o dataset de treino
    quanto para o dataset de teste e validação

    :return: dataset convertindo ao objeto python.datasets.supervised.SupervisedDataSet
    """

    dataset = ClassificationDataSet ( 4, 1 )

    for i in range ( dadosTemporario.getLength() ) :

       dataset.addSample ( dadosTemporario.getSample ( i )[0],
                           dadosTemporario.getSample ( i )[1] )

    return dataset

def montaDataset (  ) :
    """
    Função que monta o dataset dos dados
    temporários do dataset

    :return: dataset montando
    """
    # carregando o dataset do iris
    # pelo sktlearn
    iris = datasets.load_iris()
    dadosEntrada, dadosSaida = iris.data, iris.target

    # criando o dataset da iris onde : terá um array de tamanho 4 como dados de entrada
    # um array de tamanho 1 como dado de saida terá
    # 3 classes para classificar
    dataset = ClassificationDataSet ( 4, 1, nb_classes = 3 )

    for i in range ( len ( dadosEntrada ) ) :

        dataset.addSample ( dadosEntrada[i], dadosSaida[i] )

    return dataset

def main (  ) :

    # carregando o dataset do iris
    dataset = montaDataset ( )

    # carregando os datasets temporários
    # separando 60% dos dados
    # para treino e 30 % para teste
    datasetTreinoTemporario, dadosRepartidosTemporario = dataset.splitWithProportion ( 0.6 )

    # carregando os datasets temporários
    # separando os dados repartidos
    # em 50 % para teste e
    # os outros 50 % para validação
    datasetTesteTemporario, datasetValidacaoTemporario = dadosRepartidosTemporario.splitWithProportion ( 0.5 )

    # montandos os datasets finais
    datasetTreino = montaDatasetConvertido ( datasetTreinoTemporario )
    datasetTeste = montaDatasetConvertido ( datasetTesteTemporario )
    datasetValidacao = montaDatasetConvertido ( datasetValidacaoTemporario )

    # definindo a estrutura da rede adpatadas ao SoftmaxLayer
    # com dimensão 4 de entrada
    # com 20 neurônios na 1 camada intermediária
    # com 2 camadas
    # com dimensão 3 na saída
    # terá como saída adaptada à classe SoftmaxLayer
    network = buildNetwork ( 4, 20, 2, 3, outclass=SoftmaxLayer )

    # convertendo os dados para o objeto ConvertToOneOfMany
    datasetTreino._convertToOneOfMany()
    datasetTeste._convertToOneOfMany()
    datasetValidacao._convertToOneOfMany()

    # criando a rede neural treinada
    # passando a estrutura da rede neural
    # dataset para treino
    neuralNetwork = BackpropTrainer ( network, dataset=datasetTreino,learningrate=0.02, momentum=0.14, verbose=True )

    # treinando a rede até ela convergir
    # e semparando os erros em um array de erro do treino
    # e um array dos erros da validação
    errosTreino, validacaoErros = neuralNetwork.trainUntilConvergence ( dataset=datasetTreino, maxEpochs=1000 )

    # criando a camada externa de teste
    outTest = network.activateOnDataset ( datasetTeste ).argmax ( axis = 1 )
    print(5*"-"+" Precisão de teste de treinamento "+5*"-"+"\n")
    print("Precisão no teste : {} %".format ( 100 - percentError ( outTest, datasetTeste["class"] ) ) )

    # criando a camada externa de validação
    outVal = network.activateOnDataset ( datasetValidacao ).argmax ( axis = 1 )
    print("\n"+5 * "-" + " Precisão de teste de treinamento " + 5 * "-" + "\n")
    print("Precisão na validação : {} %".format ( 100 - percentError(outVal, datasetTeste["class"])))

    # fazendo o plot gráfico dos erros
    plt.plot ( errosTreino, "b", validacaoErros, "r" )
    plt.show()

    # validando os dados

if __name__ == '__main__':

    main()