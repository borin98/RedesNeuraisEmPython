""""

    Fazendo uma rede de camada dupla
    de uma rede perceptron

    Na qual os dados de entrada serão valores 0 ou 1 e iremos
    nos basear na tabela verdade da lógica matemática

    0 e 1 = 0
    0 e 0 = 0
    1 e 0 = 0
    1 e 1 = 1

"""

from CamadasOcultas import RedePerceptronDeNCamadas
import numpy as np
import random as r

def montaListaPesos ( nNeuronios ) :
    """
    Função que monta os pesos da rede neural
    No qual será a soma dos elemetos da primeira entrada e multiplicado por 0.1

    :param listaValores: lista com os valores de entrada da rede
    :return: lista de pesos
    """

    matrix = [[r.uniform ( -1.0, 1.0 ) for j in range ( 0, 2 ) ]
              for i in range ( nNeuronios ) ]

    pesoFinal = [r.uniform ( -1.0, 1.0 ) for j in range ( 0, nNeuronios )]

    return matrix, pesoFinal

def montaListaValores ( tam ) :
    """
    Função na qual monta e retorna o valor do vetor v

    :param tam: tamanho da lista
    :return: retorna a lista com os valores de entrada da rede perceptron
    """
    matrix = [ [0 for j in range ( 0, 2 ) ] for i in range ( tam )  ]

    for i in range ( 0, tam ) :

        print(5*"="+" valor "+ str( i + 1 )+" "+5*"="+"\n")

        valor = float(input("Digite o valor do primeiro elemento sendo ou 0 ou 1 : ").format ( (i + 1) ) )

        while ( (valor != 0 and valor != 1) ) :

            print("\nValor {} é inválido !!!\n".format(valor))
            valor = float(input("Digite o valor do primeiro elemento sendo ou 0 ou 1 : ").format((i + 1)))

        print("\n")
        valor2 = float(input("Digite o valor do segundo elemento sendo ou 0 ou 1 : ").format ( (i + 1) ) )
        print("\n")

        while ( ( valor2 != 0 and valor2 != 1 ) ) :

           print("\nValor {} é inválido !!!\n".format(valor2))
           valor2 = float(input("Digite o valor do segundo elemento sendo ou 0 ou 1 : ").format((i + 1)))


        matrix[i][0] = valor
        matrix[i][1] = valor2


    return matrix

def saida ( entrada ) :
    """

    Função que cria o vetor saída esperada a partir da matriz entrada

    :param entrada: matriz binária com os elementos
    :return: retorna o vetor resposta
    """

    aux = []

    for i in range ( 0, len ( entrada ) ) :

        if ( entrada[i][0] == 1 and entrada[i][1] == 1 ) :

            aux.insert (i, 1)

        else :

            aux.insert ( i, 0 )

    return aux

def acertos ( saidas, perceptron ) :
    """

    Função que retorna os acertos da rede

    :return:
    """
    acertos = 0


    for i in range ( len ( saidas ) ) :

        if ( saidas[i] == perceptron.resultado[i] ) :
            acertos += 1

    print("Porcentagem de acertos : {} %".format ( 100*( acertos/len ( saidas ) ) ) )

    return

def main (  ) :

    tam = int ( input("Digite o tamanho do vetor : ") )

    while ( tam <= 0 ) :

        print("Valor inválido !!!!")
        tam = int(input("Digite o tamanho do vetor : "))


    nNeuronios = int ( input (" Digite a quantidade de neurônios : ") )

    while ( tam <= 0 ) :

        print("Valor inválido !!!!")
        nNeuronios = int(input("Digite o tamanho do vetor : "))

    nEpocas = int(input ( " Digite a quantidade máxima de épocas : "))

    while ( nEpocas <= 0 ):

        print("Valor inválido !!!!")
        nEpocas = int(input(" Digite a quantidade máxima de épocas : "))

    listaValores = montaListaValores ( tam )
    listaPesos, pesoFinal = montaListaPesos ( nNeuronios )
    saidaEsperada = saida ( listaValores )

    # criando os numpys arrays
    entradas = np.array ( listaValores )
    pesos = np.array ( listaPesos )
    pesoFinal = np.array ( pesoFinal )
    saidas = np.array ( saidaEsperada )

    print("pesos : {}".format(pesos))

    perceptron = RedePerceptronDeNCamadas.Perceptron(entradas, saidas, pesos, pesoFinal, 0.001, nNeuronios, nEpocas)
    perceptron.treinar()

main()