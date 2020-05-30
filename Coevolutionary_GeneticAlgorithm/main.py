from ag import AG
from time import time

def main():
    
    parametros = {
        "taxa_mutacao": 0.03,
        "taxa_cruzamento": 0.9,
        "size_pop": 100,
        "stop_gen": 200,
        "num_pops": 3
    }
    
    algoritmo_genetico = AG(**parametros)
    solucao = algoritmo_genetico.start()
    print(solucao)
    algoritmo_genetico.plotaGrafico()
    
    
    
if __name__ == '__main__':
    start = time()
    main()
    print(f'Tempo de execução: {(time() - start) / 60:.2f} min')
