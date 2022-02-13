import pygad
import numpy as np
import pandas as pd
from datetime import date
import matplotlib.pyplot as plt
from portfolio import Portfolio_Analyzer
from timeit import default_timer as timer

class Optmizer():
    def __init__(
        self,
        symbols,
        parameters=None,
    ):
        self.symbols = symbols
        self.__has_runned = False
        self.filename = 'opt_results/genetic' # The filename to which the instance is saved. The name is without extension.
        last_fitness = 0
        self.parameters = parameters if parameters else {}
        self.results = {}

    def run(self, assets_data=pd.DataFrame([])):
        global last_fitness
        global fitness_func
        global on_generation

        # Defining parameters
        wallet = 1000

        pygad.random.seed(1)
        pygad.numpy.random.seed(1)
        analyzer = Portfolio_Analyzer()
        start_date = date(2021, 1, 1)
        end_date = date.today()
        years_diff = (end_date - start_date).days / 365

        if assets_data.empty:
            assets_data = analyzer.get_price_data(
                    self.symbols,
                    self.parameters['start_date'],
                    self.parameters['end_date']
            )

        rf = self.parameters['risk_free_rate'] if self.parameters['risk_free_rate'] else 0.0697

        def fitness_func(solution, solution_idx):
            if np.sum(solution) < 0 or np.sum(solution) > 1:
                return 0
            if any([ perc < 0.1 for perc in solution ]):
                return 0

            def allocation(row):
                return sum([ value*solution[i] for i, value in enumerate(row) ])

            daily_return = assets_data.pct_change(1).dropna()
            daily_return['Portfolio'] = daily_return.apply(allocation, axis=1) 

            num_days = 255 * years_diff if years_diff else 255

            return analyzer.f_sharpe_ratio(daily_return['Portfolio'], num_days, rf)

        def on_generation(ga_instance):
            global last_fitness
            print("Generation = {generation}".format(generation=ga_instance.generations_completed))
            print("Fitness    = {fitness}".format(fitness=ga_instance.best_solution(pop_fitness=ga_instance.last_generation_fitness)[1]))
            print("Change     = {change}".format(change=ga_instance.best_solution(pop_fitness=ga_instance.last_generation_fitness)[1] - last_fitness))
            last_fitness = ga_instance.best_solution(pop_fitness=ga_instance.last_generation_fitness)[1]

        num_generations = 1000 # Number of generations.
        num_parents_mating = 15 # Number of solutions to be selected as parents in the mating pool.
        sol_per_pop = 40 # Number of solutions in the population.
        num_genes = len(self.symbols)
        init_low_range = 0.0
        init_high_range = 1.0
        random_mutation_min_val = 0.0
        random_mutation_max_val = 1.0
        gene_space = { 'low': 0, 'high': 1, 'step': 0.0001 }
        mutation_type = 'random'

        ga_instance = pygad.GA(
            gene_space=gene_space,
            num_generations= self.parameters['num_generations'] if self.parameters['num_generations'] else num_generations,
            num_parents_mating = self.parameters['num_parents_mating'] if self.parameters['num_parents_mating'] else num_parents_mating,
            sol_per_pop = self.parameters['sol_per_pop'] if self.parameters['sol_per_pop'] else sol_per_pop,
            num_genes = self.parameters['num_genes'] if self.parameters['num_genes'] else num_genes,
            init_range_low=init_low_range,
            init_range_high=init_high_range,
            random_mutation_min_val=random_mutation_min_val,
            random_mutation_max_val=random_mutation_max_val,
            mutation_type=mutation_type,
            fitness_func=fitness_func,
            # on_generation=on_generation
        )

        # Running the GA to optimize the parameters of the function.
        start_time = timer()
        ga_instance.run()
        end_time = timer()
        self.results['time_elapsed'] = end_time - start_time
        
        fig, ax = plt.subplots(figsize=(15, 8))
        ga_instance.plot_fitness().savefig('graphs/optimization_fitness.png')
        self.results['fitness_graph'] = fig


        # Returning the details of the best solution.
        solution, solution_fitness, solution_idx = ga_instance.best_solution(ga_instance.last_generation_fitness)
        print("Parameters of the best solution : {solution}".format(solution=solution))
        print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=solution_fitness))
        print("Sum of the best solution: {solution_sum}".format(solution_sum=np.sum(solution)))
        print("Index of the best solution : {solution_idx}".format(solution_idx=solution_idx))

        if ga_instance.best_solution_generation != -1:
            print("Best fitness value reached after {best_solution_generation} generations.".format(best_solution_generation=ga_instance.best_solution_generation))

        self.results['solution'] = solution
        self.results['solution_fitness'] = solution_fitness
        self.results['solution_idx'] = solution_idx
        self.results['ga_instance'] = ga_instance
        self.__has_runned = True

        solution = {
            assets_data.columns.values[i]: percentage for i, percentage in enumerate(solution)
        }
        return solution

        def save_instance(self):
            # Saving the GA instance.
            ga_instance.save(filename=self.filename)

        def load_instance(self):
            # Loading the saved GA instance.
            loaded_ga_instance = pygad.load(filename=self.filename)
            loaded_ga_instance.plot_fitness()
