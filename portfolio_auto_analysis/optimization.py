import numpy as np
import pygad
from datetime import date
import matplotlib.pyplot as plt
from . import Portfolio_Analyzer

last_fitness = 0

def sharpe_ratio(return_series, N, risk_free=0):
    mean = return_series.mean() * N  -risk_free
    sigma = return_series.std() * np.sqrt(N)
    return mean / sigma

def sortino_ratio(return_series, N, risk_free=0):
    mean = return_series.mean() * N -risk_free
    std_neg = return_series[return_series < 0].std() * np.sqrt(N)
    return mean / std_neg

def example():
    global last_fitness
    global fitness_func
    global on_generation

    # Defining parameters
    wallet = 1000
    stocksymbols = [
        'AAPL',
        'AMZN',
        'MSFT',
        'GOOGL',
        'FB'
    ]

    analyzer = Portfolio_Analyzer()

    start_date = date(2013, 1, 1)
    end_date = date(2020, 10, 1)

    work_days = np.busday_count(start_date, end_date)

    data_frame = analyzer.get_price_data(
        stocksymbols,
        start_date,
        end_date
    )

    # portf_table = analyzer.portfolio_table(
        # data_frame,
        # [ 1/len(stocksymbols) for _ in stocksymbols ],
        # wallet
    # )

    def fitness_func(solution, solution_idx):
        if np.sum(solution) < 0 or np.sum(solution) > 1:
            return 0
       
        def allocation(row):
            return sum([ value*solution[i] for i, value in enumerate(row) ])

        # portf_table = analyzer.portfolio_table(
            # data_frame,
            # solution,
            # wallet
        # )

        daily_return = data_frame.pct_change(1).dropna()
        daily_return['Portfolio'] = daily_return.apply(allocation, axis=1) 

        sr = sortino_ratio(daily_return['Portfolio'], 255, 0.01)

        # daily_return = portf_table['Total Pos'].pct_change(1)
        # sharpe_ratio = (daily_return.mean() / daily_return.std()) * np.sqrt(255)

        # daily_simple_return = analyzer.periodic_simple_returns(
            # data_frame,
            # 1
        # )

        # avg_daily = analyzer.average_PSR(daily_simple_return)

        # standard_deviation = analyzer.annualized_standard_deviation(
            # daily_simple_return,
            # work_days
        # )

        # sharpe_ratio = analyzer.sharpe_ratio(avg_daily, standard_deviation)
        # fitness = 0

        # for i, percentage in enumerate(solution):
            # fitness += percentage*avg_daily[i]

        return sr

    def on_generation(ga_instance):
        global last_fitness
        print("Generation = {generation}".format(generation=ga_instance.generations_completed))
        print("Fitness    = {fitness}".format(fitness=ga_instance.best_solution(pop_fitness=ga_instance.last_generation_fitness)[1]))
        print("Change     = {change}".format(change=ga_instance.best_solution(pop_fitness=ga_instance.last_generation_fitness)[1] - last_fitness))
        last_fitness = ga_instance.best_solution(pop_fitness=ga_instance.last_generation_fitness)[1]

    num_generations = 300 # Number of generations.
    num_parents_mating = 15 # Number of solutions to be selected as parents in the mating pool.
    sol_per_pop = 40 # Number of solutions in the population.
    num_genes = len(stocksymbols)
    init_low_range = 0.0
    init_high_range = 1.0
    random_mutation_min_val = 0.0
    random_mutation_max_val = 1.0
    initial_population = [ [ 1/len(stocksymbols) for _ in stocksymbols ] for _ in range(sol_per_pop) ]
    gene_space = np.linspace(0, 1, 300)

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
        fitness_func=fitness_func,
        on_generation=on_generation
    )

    # Running the GA to optimize the parameters of the function.
    ga_instance.run()
    ga_instance.plot_fitness().savefig('graphs/optimization_fitness.png')


    # Returning the details of the best solution.
    solution, solution_fitness, solution_idx = ga_instance.best_solution(ga_instance.last_generation_fitness)
    print("Parameters of the best solution : {solution}".format(solution=solution))
    print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=solution_fitness))
    print("Index of the best solution : {solution_idx}".format(solution_idx=solution_idx))

    if ga_instance.best_solution_generation != -1:
        print("Best fitness value reached after {best_solution_generation} generations.".format(best_solution_generation=ga_instance.best_solution_generation))

    # Saving the GA instance.
    filename = 'opt_results/genetic' # The filename to which the instance is saved. The name is without extension.
    ga_instance.save(filename=filename)

    # Loading the saved GA instance.
    loaded_ga_instance = pygad.load(filename=filename)
    loaded_ga_instance.plot_fitness()
