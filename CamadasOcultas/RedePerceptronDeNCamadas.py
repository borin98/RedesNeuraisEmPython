"""

    Objeto que cria a rede neural de

    Reajustar os pesos da rede

    Problemas encontrados :
    O problema está na linha 544 do código

"""

from FuncoesDeAtivacao import Sigmoide
import numpy as np

class Perceptron:

    def __init__(self, listaValoresEntrada, listaValoresSaida, listaPesos, pesoFinal, taxaAprendizado=0.1, nNeuronios=1,
                 nEpocas=1000, momento=0.001):

        self.sig = Sigmoide.Sigmoide
        self.listaValoresEntrada = listaValoresEntrada
        self.listaValoresSaida = listaValoresSaida
        self.listaPesos = listaPesos
        self.pesoFinal = pesoFinal
        self.ativacao = False
        self.taxaAprendizado = taxaAprendizado
        self.erroTotal = 1
        self.resultado = []
        self.nNeuronios = nNeuronios
        self.pesosSaida = np.zeros(self.nNeuronios)
        self.nEpocas = nEpocas
        self.saidaCamadasIntermediarias = np.array([0 for i in range(nNeuronios)])
        self.momento = momento
        print("Lista inicial da camada de saida intermediária : {}".format(self.saidaCamadasIntermediarias))

    def backPropagation(self):
        """
        Faz o backPropagation da rede

        :return:
        """

    def calculaSaidaCamadaIntermediaria(self):
        """
        Faz o cálculo da saída final da rede

        :param valorEntrada:
        :return:
        """

        print("Saidas Camadas Intermediárias : {}\n".format(self.saidaCamadasIntermediarias))
        print("Pesos Saída : {}".format ( self.pesoFinal ) )
        valor = self.saidaCamadasIntermediarias.dot(self.pesosSaida)
        sigmoide =  self.sig(valor)
        return sigmoide.valorDerivada

    def calculaSaida(self, valorEntrada, indice):
        """

        Função que faz a soma da função de ativação

        :return: se a função foi ativada ou não
        """

        # self.pesosSaida[indice] = r.randint ( -1, 1 )

        lista = []
        s = 0

        print ( "Valor entrada : {}".format( valorEntrada ) )

        for neuronios in range(self.nNeuronios):
            valor = (valorEntrada[0] * self.listaPesos[neuronios][0]) + (
                        valorEntrada[1] * self.listaPesos[neuronios][1])

            lista.insert(neuronios, valor)

        numpyValor = np.array(lista)

        s = numpyValor.dot(numpyValor)

        print(s)

        sigmoide = self.sig(s)
        # self.atualizaPeso ( sigmoide )
        return sigmoide.funcaoSigmoide()

    def atualizaPeso(self, delta):
        """

        Função que atualiza os pesos da rede neural ao longo do tempo

        :param i: entrada i da função treinar
        :return:
        """
        # self.listaPesos[j] = sigmoide.valorDerivada

        print("Saida Camada Intermediária : {}".format(self.saidaCamadasIntermediarias))

        for i in range( 0, len(self.listaPesos) ) :

            self.pesoFinal[i] = ( self.pesoFinal[i]*self.momento ) + ( self.saidaCamadasIntermediarias[i] *delta*self.taxaAprendizado )

        print("Peso atualizado : {}".format(self.listaPesos))

    def montaResposta(self):
        """

        Função que monta a resposta final da rede neural

        :return:
        """

        for i in range(len(self.listaValoresEntrada)):
            saidaCalculada = self.calculaSaida(np.asanyarray(self.listaValoresEntrada[i], i))
            self.resultado.insert(i, saidaCalculada)

    def treinar(self):
        """

        Função que faz o treino da rede

        :return:
        """
        epocasTotais = 0

        while (self.erroTotal != 0 or
               epocasTotais <= self.nEpocas):

            self.erroTotal = 0

            print("Epoca {}\n".format(self.nEpocas))

            for tam in range(0, len(self.listaValoresSaida)):

                # percorre todos os neurônios da camada escondida
                # realizando o cálculo das saídas das camadas intermediárias
                for i in range(self.nNeuronios):
                    valorEntrada = np.asanyarray(self.listaValoresEntrada[i])
                    saidaCalculada = self.calculaSaida(valorEntrada, i)
                    print("Saida Calculada : {}".format ( saidaCalculada ) )
                    self.saidaCamadasIntermediarias[i] = saidaCalculada

                saidaCalculada = self.calculaSaidaCamadaIntermediaria()
                print("Saída calculada : {}".format(saidaCalculada))
                print("lista valores saída : {} ".format(self.listaValoresSaida))
                print("Saída camadas intermediárias : {}".format(self.saidaCamadasIntermediarias))

                # fazendo o cálulo final da rede
                valorEstimado = self.saidaCamadasIntermediarias.dot ( self.pesoFinal )
                print("Valor estimado : {}".format(valorEstimado))

                if ( valorEstimado != self.listaValoresSaida[tam]):
                    erro = self.listaValoresSaida[tam] - saidaCalculada
                    sig = self.sig(erro)
                    deltaSaida = erro * sig.derivadaSigmoide()
                    print("Delta saída : {}".format(deltaSaida))
                    deltaEscondida = sig.derivadaSigmoide() * self.pesoFinal[tam] * deltaSaida
                    print("delta escondido : {}".format(deltaEscondida))

                    # fazendo a autalização dos pesos
                    self.atualizaPeso( delta=deltaEscondida )

            epocasTotais += 1

        print("Total de erros : {} %".format(self.erroTotal))

        self.montaResposta()
