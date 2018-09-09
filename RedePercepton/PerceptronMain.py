from RedePercepton import Perceptron

"""

Implementação de uma rede perceptro

"""

def main (  ) :

    # dados criados para teste
    amostra = [
            [ 0, 0 ] ,
            [ 0, 1 ] ,
            [ 1, 0 ] ,
            [ 1, 1 ]
    ]

    saida = [ 0, 1, 1, 1 ]

    rede = Perceptron.Perceptron ( amostra, saida )
    rede.train (  )
    rede.test ( [0, 1] )

main()