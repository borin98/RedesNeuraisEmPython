import numpy as np

class Sigmoide :

    def derivadaSigmoide(self) :
        """
        Função de derivada da sigmoide

        :return:
        """
        return self.saida * ( 1 - self.saida )

    def funcaoSigmoide ( self ) :
        """

        Função que faz a implementação da sigmoide
        e retorna o valor dela ao correspondende valor de entrada

        :return: valor da sigmoide em relação ao valor de entrada
        """

        valor = 1/( 1 + np.exp ( -self.entrada ) )
        return valor

    def __init__(self, entrada ) :

        self.entrada = entrada
        self.saida = self.funcaoSigmoide()
        self.valorDerivada = self.derivadaSigmoide()

