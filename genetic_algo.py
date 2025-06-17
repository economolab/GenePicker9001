# -*- coding: utf-8 -*-
"""
Genetic algorithm functions
"""

import logging
import pygad
import random

import matplotlib.pyplot as plt

from sklearn.manifold import MDS
from sklearn.metrics import pairwise_distances

import numpy as np
import pandas as pd

from ABC_toolbox import classify_cells

# %%

# build initial population
# goal is to make sure each gene shows up in at least 2 solutions, while having
# the smallest possible number of solutions otherwise
def gen_init_pop(sol_size, copies, top_n):

    init_pop = []
    
    for _ in range(copies):
        inds = list(range(top_n))
        
        random.shuffle(inds)
        
        chunks = [inds[x:x+sol_size] for x in range(0, len(inds), sol_size)]
        
        while len(chunks[-1]) != sol_size:
            
            extras_needed = sol_size - len(chunks[-1])
            extras = random.sample(inds, extras_needed)
            
            for extra in extras:
                if extra not in chunks[-1]:
                    chunks[-1].append(extra)
                    
        init_pop.extend(chunks)
        
    return init_pop

def ind2gene(inds, gene_names):

    gene_subset = [gene_names[ind] for ind in inds]
        
    return gene_subset

def gene2ind(gene, gene_names):
    
    ind = np.where(gene_names == gene)[0][0]
    
    return ind

def fitness_func(ga_instance, solution, solution_idx):
    
    gene_subset = ind2gene(solution, ga_instance.gene_names)
    
    if ga_instance.set_genes is not None:
        set_genes = [str(el) for el in ga_instance.set_genes]
        gene_subset.extend(set_genes)
    
    res = classify_cells.cross_val_classifier(ga_instance.meta, 
                                              ga_instance.exp, 
                                              ga_instance.freqs, 
                                              genes=gene_subset,
                                              verbose=False)
    
    acc = np.mean(res["accs"])
    sparsity = np.mean(res["sparsities"])
    
    solution_fitness = (1/3)*acc + (2/3)*sparsity
    
    return solution_fitness

# get average performance per gene and number of solutions it was evaluated in
def collate_ga_results(ga_instance, top_genes):
    
    solutions = ga_instance.solutions
    solutions_fitness = ga_instance.solutions_fitness
    
    gene_occurences = []
    gene_fitnesses = []
    
    for top_gene in top_genes:
        
        ind = gene2ind(top_gene, top_genes)
        occurences = 0
        fitnesses = []
        
        for i in range(len(solutions)):
            if ind in solutions[i]:
                occurences += 1
                fitnesses.append(solutions_fitness[i])
        
        gene_occurences.append(occurences)
        gene_fitnesses.append(fitnesses)
        
    gene_fitnesses = [np.mean(el) for el in gene_fitnesses]
        
    gene_df_ga = {}
    gene_df_ga['gene'] = top_genes
    gene_df_ga['occurences'] = gene_occurences
    gene_df_ga['fitnesses'] = gene_fitnesses
    gene_df_ga = pd.DataFrame(data=gene_df_ga)
    
    sol_df_ga = {}
    sol_df_ga['solution'] = [ind2gene(el, top_genes) for el in solutions]
    sol_df_ga['fitness'] = solutions_fitness
    sol_df_ga = pd.DataFrame(data=sol_df_ga)
    
    return gene_df_ga, sol_df_ga

def build_logger():
    
    level = logging.DEBUG
    name = 'logfile.txt'
    
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    file_handler = logging.FileHandler(name,'a+','utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_format = logging.Formatter('%(asctime)s %(levelname)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    file_handler.setFormatter(file_format)
    logger.addHandler(file_handler)
    
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_format = logging.Formatter('%(message)s')
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)
    
    return logger

def run_ga(meta, exp, freqs, top_genes, 
            init_copies=2,
            num_generations=100,
            num_genes=12,
            parent_selection_type='rws',
            crossover_probability=0.5,
            mutation_type='adaptive',
            mutation_probability=[0.5, 0],
            set_genes=None):
    
    if set_genes is not None:
        
        top_genes = list(top_genes)
        
        for set_gene in set_genes:
            if set_gene in top_genes:
                top_genes.remove(set_gene)
                
        num_genes = num_genes - len(set_genes)
    
    top_genes = np.array(top_genes)
    
    gene_type = int
    keep_elitism = 1
    crossover_type = 'uniform'
    mutation_by_replacement=True
    gene_space = np.arange(len(top_genes))
    save_best_solutions = True
    save_solutions = True
    allow_duplicate_genes = False
    
    initial_population = gen_init_pop(num_genes, init_copies, len(top_genes))
    num_parents_mating = np.floor(len(initial_population) / 2)
    num_parents_mating = int(num_parents_mating)
    
    logger = build_logger()
    
    def on_generation(ga_instance):
        ga_instance.logger.info(f"Generation = {ga_instance.generations_completed}")
        ga_instance.logger.info(f"Fitness    = {ga_instance.best_solution(pop_fitness=ga_instance.last_generation_fitness)[1]}")
    
    ga_instance = pygad.GA(num_generations=num_generations,
                           num_parents_mating=num_parents_mating,
                           fitness_func=fitness_func,
                           initial_population=initial_population,
                           num_genes=num_genes,
                           gene_type=gene_type,
                           parent_selection_type=parent_selection_type,
                           keep_elitism=keep_elitism,
                           crossover_type=crossover_type,
                           crossover_probability=crossover_probability,
                           mutation_type=mutation_type,
                           mutation_probability=mutation_probability,
                           mutation_by_replacement=mutation_by_replacement,
                           gene_space=gene_space,
                           save_best_solutions=save_best_solutions,
                           save_solutions=save_solutions,
                           suppress_warnings=True,
                           allow_duplicate_genes=allow_duplicate_genes,
                           on_generation=on_generation,
                           logger=logger)
    
    ga_instance.meta = meta
    ga_instance.exp = exp
    ga_instance.freqs = freqs
    ga_instance.gene_names = top_genes
    ga_instance.set_genes = set_genes
    
    ga_instance.run()
    
    gene_df_ga, sol_df_ga = collate_ga_results(ga_instance, top_genes)
    
    return gene_df_ga, sol_df_ga, ga_instance

def jaccard_distance(A, B):
    
    A = set(A)
    B = set(B)
    
    num = len(A.symmetric_difference(B))
    denom = len(A.union(B))
    
    d = num / denom
    
    return d

def plot_ga(sol_df_ga):

    solutions = sol_df_ga["solution"].values
    
    print("Calculating distances...")
    D = pairwise_distances(np.vstack(solutions), metric=jaccard_distance)

    print("Embedding using MDS...")
    embedding = MDS(n_components=2, dissimilarity='precomputed')
    X_transformed = embedding.fit_transform(D)
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    ax.plot_trisurf(X_transformed[:,0], X_transformed[:,1], sol_df_ga['fitness'].values,
                    cmap=plt.cm.CMRmap)
        
