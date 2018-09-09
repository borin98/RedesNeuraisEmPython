import matplotlib.pyplot as plt
import random as r
"""

Classe na qual cria o objeto Perceptron para a análise de seu uso
no algorítmo principal

"""

class Perceptron :

    def __init__(self, amostras, saidas, taxaAprendizado=0.1, epocas = 1000, limiar = -1) :

        self.amostras = amostras
        self.saidas = saidas
        self.taxaAprendizado = taxaAprendizado
        self.epocas = epocas
        self.limiar = limiar
        self.nAmostras = len ( amostras )
        self.nAtributos = len ( amostras[0] )
        self.pesos = []

    def sinal(self, u) :
        """

        Função sinal como parâmetro

        :param u:
        :return:
        """

        if (u >= 0):
            return 1

        return -1


    def test(self, amostra ) :
        """
        Função que faz o teste da classificação
        dos dados

        :param amostra:
        :param classe1:
        :param classe2:
        :return:
        """

        amostra.insert ( 0, -1 )

        u = 0

        for i in range ( self.nAtributos + 1 ) :

            u += self.pesos [ i ] * amostra [ i ]

        y = self.degrau ( u )

        if ( y == 0 ) :

            print("Classe 1 para o dado {}\n\n".format ( y ) )

            return

        print("Classe 2 para o dado {}\n\n".format(y))

    def degrau(self, u) :
        """
        Função degrau simples que retorna 1 se u >= 0
        e 0 caso contrário

        :param u:
        :return:
        """

        if ( u >= 0 ) :

            return 1

        return 0

    def train(self) :
        """

        Função que faz o treinamento da rede neural

        :return:
        """

        for amostra in self.amostras :

            amostra.insert ( 0, -1 )

        for i in range ( self.nAtributos ) :

            self.pesos.append ( r.random (  ) )

        self.pesos.insert ( 0, self.limiar )

        Nepocas = 0 # contandor de épocas

        while True :

            erro = False
            pass

            for i in range ( self.nAmostras ) :

                u = 0

                for j in range ( self.nAtributos + 1 ) :

                    u += self.pesos[j]*self.amostras[i][j]

                y = self.degrau ( u )   # dado de saída da rede

                # verifica se a saída da rede é diferente da saída desejada
                if ( y != self.saidas[i] ) :

                    erroAux = self.saidas [ i ] - y

                    # fazendo o ajuste dos pesos para cada elemento da amostra
                    for k in range ( self.nAtributos + 1 ) :

                        self.pesos [ k ] += self.taxaAprendizado * erroAux *self.amostras [ i ] [ k ]

                    erro = True # mostrando que o erro ainda existe, continua o looping

            Nepocas += 1

            # condição de parada do algorítmo
            if ( not erro or
                Nepocas > self.epocas) :

                break

            if ( not erro ) :
                break