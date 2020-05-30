from individuo import Individuo
from random import sample, randrange, random, uniform
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.lines as mlines

from os import path
class AG:
    def __init__(self, **parametros):
        self.taxa_mutacao = parametros["taxa_mutacao"]
        self.taxa_cruzamento = parametros["taxa_cruzamento"]
        self.size_pop = parametros["size_pop"]
        self.stop_gen = parametros["stop_gen"]
        self.melhor_solucao = None
        self.quantidade_de_populacoes = parametros["num_pops"]
        
    
    def start(self,):
        self.resetData()
        populacoes = [{"populacao": self.gerarPopInicial(), "melhor_individuo": None} for _ in range(self.quantidade_de_populacoes)]
        for pop in populacoes:
            self.avaliar(pop["populacao"])
            self.encontraMelhor(pop)
            self.salvaDados(pop["populacao"])
        for geracao in range(1, self.stop_gen):
            self.compartilhaConhecimento(populacoes)
            populacoes = [{"populacao": self.reproducao(pop["populacao"], geracao), "melhor_individuo": pop["melhor_individuo"]} for pop in populacoes]
            for pop in populacoes:
                self.avaliar(pop["populacao"])
                self.encontraMelhor(pop)
                self.salvaDados(pop["populacao"])
        
            
        #self.resetData()
        return self.melhor_solucao
    
    def gerarPopInicial(self,)->list:
        return [Individuo() for _ in range(self.size_pop)]
    
    
    def avaliar(self, populacao:list):
        for individuo in populacao:
            individuo.fitness = individuo.calcFitness()
            
        
    def encontraMelhor(self, populacao:dict):
        populacao["populacao"].sort(key=lambda individuo: individuo.fitness)
        melhor = populacao["populacao"][0]
        
        if not populacao["melhor_individuo"]:
            populacao["melhor_individuo"] = melhor.copia()
    
        if melhor.fitness < populacao["melhor_individuo"].fitness:
            populacao["melhor_individuo"] = melhor.copia()
        
        if not self.melhor_solucao:
            self.melhor_solucao = melhor.copia()
        
        if melhor.fitness < self.melhor_solucao.fitness:
            self.melhor_solucao = melhor.copia()
        
        
    def reproducao(self, populacao:list, geracao)->list:
        piscina = self.selecao(populacao)
        nova_populacao = self.cruzamento(piscina, geracao)
        self.mutacao(nova_populacao)
        nova_populacao.sort(key=lambda individuo: individuo.fitness)
        percentual = int(self.size_pop * self.taxa_cruzamento)
        percentual = percentual if percentual % 2 == 0 else percentual + 1
        populacao.sort(key = lambda indv: indv.fitness, reverse=True)
        
        return nova_populacao + populacao[percentual: ]
    
    def selecao(self, populacao:list)->list:
        piscina = []
        quantidade = 3
        percentual = int(self.size_pop * self.taxa_cruzamento)
        percentual = percentual if percentual % 2 == 0 else percentual + 1
        
        for _ in range(percentual):
            selecionados = sample(populacao, quantidade)
            selecionados.sort(key=lambda individuo: individuo.fitness)
            melhor = selecionados[0]
            piscina.append(melhor)
        
        return piscina
    
    def cruzamento(self, piscina:list, geracao:int)->list:
        nova_populacao = []
        percentual = int(self.size_pop * self.taxa_cruzamento)
        percentual = percentual if percentual % 2 == 0 else percentual + 1
        size = len(piscina)
        for _ in range(percentual):
            indv = piscina[randrange(size)]
            indv2 = piscina[randrange(size)]
            filho1, filho2 = self.cruzamentoUmPonto(indv.cromossomo, indv2.cromossomo)
            nova_populacao.append(Individuo(cromossomo=filho1, geracao=geracao))
            nova_populacao.append(Individuo(cromossomo=filho2, geracao=geracao))
            
        return nova_populacao
    
    def cruzamentoUmPonto(self, seq1:str, seq2:str)->tuple:
        p_seq1 = randrange(len(seq1))
        p_seq2 = p_seq1
        
        seq12 = seq1[ :p_seq1] + seq2[p_seq2: ]
        seq21 = seq2[ :p_seq2] + seq1[p_seq1: ]
        
        return(seq12, seq21)
    
    def cruzamentoDoisPontos(self, seq1:str, seq2:str)->tuple:
        p_seq1 = randrange(len(seq1))
        p_seq2 = p_seq1
        
        seq12 = seq1[ :p_seq1] + seq2[p_seq1 :p_seq2] + seq1[p_seq2: ]
        seq21 = seq2[ :p_seq1] + seq1[p_seq1 :p_seq2] + seq2[p_seq2: ]
        
        return(seq12, seq21)
    
    def mutacao(self, populacao):
        for indiv in populacao:
            mutate = random() < self.taxa_mutacao
            if mutate:
                size = len(indiv.cromossomo)
                n1, n2 = randrange(size), randrange(size)
                #posicao = randrange(size)
                #indiv.cromossomo[posicao]= uniform(-10, 10)
                indiv.cromossomo = indiv.cromossomo[:n1] + [indiv.cromossomo[n2]] + \
                indiv.cromossomo[n1+1:n2] + [indiv.cromossomo[n1]] + indiv.cromossomo[n2+1: ]
                    
    
    
    def compartilhaConhecimento(self, populacoes:dict):
        for pop in populacoes:
            for vizinho in populacoes:
                if pop != vizinho:
                    pop["populacao"].append(vizinho["melhor_individuo"].copia())
                    pop["populacao"].pop(0)
    
    def salvaDados(self, populacao):
        todos_fitness = np.array([individuo.fitness for individuo in populacao])
        with open(path.abspath('dados.npy'), "ab+") as file:
            np.save(file, todos_fitness)
            
            
    def plotaGrafico(self, ):
        geracoes = np.arange(self.stop_gen)
        melhores = np.ndarray((0))
        piores = np.ndarray((0))
        media = np.ndarray((0))
        with open(path.abspath('dados.npy'),  "rb") as file:
            for _ in range(self.stop_gen):
                    all_fitness = np.load(file)
                    melhores = np.append(melhores, min(all_fitness))
                    piores = np.append(piores, max(all_fitness))
                    media = np.append(media, all_fitness.mean())
        
        label = ["melhores", "piores", "média"]
        data = [melhores, piores, media]
        
        plt.style.use('ggplot')
        
        for l,y in zip(label, data):
            plt.plot(geracoes, y, label=l)
        
        plt.title("Relação Fitness/Geração")
        plt.xlabel("Geração")
        plt.ylabel("Fitness")
        plt.legend(loc="best")
        plt.grid(True) 
        plt.show()
        
            
    
    def resetData(self, ):
        open(path.abspath('dados.npy'),"wb").close()