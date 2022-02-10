import pygad
import numpy as np
from datetime import date
import matplotlib.pyplot as plt
from . import Portfolio_Analyzer

class Optmizer():
    def __init__(self):
        self.__has_runned = False
        self.solution = []
        self.filename = 'opt_results/genetic' # The filename to which the instance is saved. The name is without extension.
        last_fitness = 0
        pass

    def run(self):
        global last_fitness
        global fitness_func
        global on_generation

        # Defining parameters
        wallet = 1000
        stocksymbols = [
            'AAPL',
            'AMZN',
            'MSFT',
            'EOG',
            'OXY',
            'HAL',
            'SLB'
        ]
        pygad.random.seed(1)
        pygad.numpy.random.seed(1)
        analyzer = Portfolio_Analyzer()
        start_date = date(2021, 1, 1)
        end_date = date.today()
        years_diff = (end_date - start_date).days / 365

        assets_data = analyzer.get_price_data(
                stocksymbols,
                start_date,
                end_date
        )

        def fitness_func(solution, solution_idx):
            if np.sum(solution) < 0 or np.sum(solution) > 1:
                return 0

            def allocation(row):
                return sum([ value*solution[i] for i, value in enumerate(row) ])
            daily_return = assets_data.pct_change(1).dropna()
            daily_return['Portfolio'] = daily_return.apply(allocation, axis=1) 
            num_days = 255 * years_diff if years_diff else 255

            return analyzer.f_sortino_ratio(daily_return['Portfolio'], num_days, 0.0697)

        def on_generation(ga_instance):
            global last_fitness
            print("Generation = {generation}".format(generation=ga_instance.generations_completed))
            print("Fitness    = {fitness}".format(fitness=ga_instance.best_solution(pop_fitness=ga_instance.last_generation_fitness)[1]))
            print("Change     = {change}".format(change=ga_instance.best_solution(pop_fitness=ga_instance.last_generation_fitness)[1] - last_fitness))
            last_fitness = ga_instance.best_solution(pop_fitness=ga_instance.last_generation_fitness)[1]

        num_generations = 2000 # Number of generations.
        num_parents_mating = 15 # Number of solutions to be selected as parents in the mating pool.
        sol_per_pop = 40 # Number of solutions in the population.
        num_genes = len(stocksymbols)
        init_low_range = 0.0
        init_high_range = 1.0
        random_mutation_min_val = 0.0
        random_mutation_max_val = 1.0
        gene_space = { 'low': 0, 'high': 1, 'step': 0.0001 }
        mutation_type = 'random'

        ga_instance = pygad.GA(
            gene_space=gene_space,
            num_generations=num_generations,
            num_parents_mating=num_parents_mating,
            sol_per_pop=sol_per_pop,
            num_genes=num_genes,
            init_range_low=init_low_range,
            init_range_high=init_high_range,
            random_mutation_min_val=random_mutation_min_val,
            random_mutation_max_val=random_mutation_max_val,
            mutation_type=mutation_type,
            fitness_func=fitness_func,
            # on_generation=on_generation
        )

        # Running the GA to optimize the parameters of the function.
        ga_instance.run()
        ga_instance.plot_fitness().savefig('graphs/optimization_fitness.png')


        # Returning the details of the best solution.
        solution, solution_fitness, solution_idx = ga_instance.best_solution(ga_instance.last_generation_fitness)
        print("Parameters of the best solution : {solution}".format(solution=solution))
        print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=solution_fitness))
        print("Sum of the best solution: {solution_sum}".format(solution_sum=np.sum(solution)))
        print("Index of the best solution : {solution_idx}".format(solution_idx=solution_idx))

        if ga_instance.best_solution_generation != -1:
            print("Best fitness value reached after {best_solution_generation} generations.".format(best_solution_generation=ga_instance.best_solution_generation))

        self.solution = solution
        self.__has_runned = True
        return solution

        def save_instance(self):
            # Saving the GA instance.
            ga_instance.save(filename=self.filename)

        def load_instance(self):
            # Loading the saved GA instance.
            loaded_ga_instance = pygad.load(filename=self.filename)
            loaded_ga_instance.plot_fitness()

