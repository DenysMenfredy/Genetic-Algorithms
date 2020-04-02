from ag import AG
from time import time

def run(iteracoes):
    parametros = {
        "taxa_mutacao": 0.05,
        "taxa_cruzamento": 0.9,
        "size_pop": 100,
        "stop_gen": 200,
        "num_pops": 5
    }

    wins = 0
    
    for _ in range(iteracoes):
        algoritmo_genetico = AG(**parametros)
        solucao = algoritmo_genetico.start()
        #print(f'Rodando iteração {i}')
        if solucao.fitness < -960 and solucao.fitness > -958:
            wins += 1
        print(solucao)
        algoritmo_genetico.plotaGrafico()

    percentual = (wins * 100) / iteracoes
    print(f'Percentual de acertos: {percentual}% ')
    
def main():
    start = time()
    run(1)
    print(f'Tempo de execução: {(time() - start) / 60:.2f} min')

   
    
if __name__ == '__main__':
    main()