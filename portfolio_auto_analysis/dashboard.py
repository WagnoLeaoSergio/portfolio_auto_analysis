import sys
import os
import pandas as pd
import streamlit as st
from datetime import datetime
from streamlit import cli as stcli
from optimization import Optmizer
from portfolio import Portfolio_Analyzer

class Dashboard():
    def start():
        st.title("Portfolio Analysis")

        df = pd.DataFrame({
            'first column': [1, 2, 3, 4],
            'second column': [10, 20, 30, 40]
        })

        st.write(df)
        sys.argv = ["streamlit", "run", __file__]
        sys.exit(stcli.main())

    def show():
        st.title("Portfolio Optimization")

        st.markdown("#### Select the assets: ")
        symbols = [
            'GBTC',
            'EXPI',
            'AMD',
            'FIV',
            'CYRX',
            'NVDA',
            'ENPH',
            'RNG',
            'APPS',
            'HAL',
            'SLB',
            'OXY',
            'EOG',
            'HES',
            'XOM',
            'APA',
            'COP',
            'PXD',
            'AMZN',
            'MSFT',
            'DISCK',
            'DVN'
        ]
        assets_symbols = st.multiselect(
            "Assets:",
            sorted(symbols)
        )

        st.markdown("#### Select the start date to make the analysys:")
        start_date = st.date_input(
            "Start Date - Note: The end date will be the current time."
        )
        end_date = datetime.today().date()

        if start_date >= end_date:
            st.error('Error! Invalid date interval.')

        dwn_data_btn = st.button('Download data')

        analyzer = Portfolio_Analyzer()
        assets_data = None

        if dwn_data_btn:
            assets_data = analyzer.get_price_data(
                symbols=assets_symbols,
                start_date=start_date,
                end_date=end_date
            )
            st.markdown('##### Downloaded assets:')
            st.table(assets_data.head())
            assets_data.to_csv('opt_results/downloaded_data.csv')    

        st.markdown('#### Capital allocation ($): ')
        allocation = st.number_input(
                'Allocation',
                min_value=0,
                value=1
        )

        st.markdown('#### Risk-Free Rate:')
        risk_free_rate = st.number_input(
                'Risk-Free',
                0.0,
                1.0,
                value=0.0697,
                step=0.01
        )

        st.header("GA Parameters")

        num_generations = int(st.number_input(
                'Number of Generations: ',
                1,
                value=50
        ))
        sol_per_pop = int(st.number_input(
                'Number of solutions in the population: ',
                1,
                value=40
        ))
        num_parents_mating = int(st.number_input(
                'Number of solutions to be selected as parents in the mating pool: ',
                1,
                max_value=int(sol_per_pop),
                value=15
        ))
        num_genes = len(assets_symbols)

        solution = None

        run_sim_btn = st.button('Run Simulation')
        if run_sim_btn:
            assets_data = pd.read_csv('opt_results/downloaded_data.csv')
            assets_data['Date'] = pd.to_datetime(assets_data['Date'])
            assets_data.set_index('Date', inplace=True)

            if assets_data.empty:
                st.error('Error! No data downloaded.')

            parameters = {
                    'start_date': start_date,
                    'end_date': end_date,
                    'num_generations': num_generations,
                    'sol_per_pop': sol_per_pop,
                    'num_parents_mating': num_parents_mating,
                    'num_genes': num_genes,
                    'risk_free_rate': risk_free_rate
            }
            optimizer = Optmizer(assets_symbols, parameters)

            with st.spinner(text='Running'):
                solution = optimizer.run(assets_data)
                st.success('Finished')

            time_elapsed = round(optimizer.results['time_elapsed'], 5)
            st.markdown(f'##### Execution Time: {time_elapsed} seconds')
            st.markdown('Solution founded:')
            st.write(solution)


        if solution:
            st.pyplot(
                    analyzer.plot_allocations(
                        solution.values(),
                        index=solution.keys()
                    )
            )

            st.header('Analysis:') 
            tables = { 
                name : pd.DataFrame(assets_data[name]) for name in \
                    assets_data.columns.values if name != 'Date'
            }

            for value in solution.keys():
                tables[value]['Norm return'] = tables[value] / tables[value].iloc[0]
                tables[value]['Allocation'] = tables[value]['Norm return'] * solution[value]
                tables[value]['Position'] = tables[value]['Allocation'] * allocation

            all_tables = [ table['Position'] for table in tables.values() ]
            portf_table = pd.concat(all_tables, axis=1)
            portf_table.columns = solution.keys()
            portf_table['Total Pos'] = portf_table.sum(axis=1)

            st.markdown('##### Portfolio table:')
            st.table(portf_table.tail())

            with st.expander('Visualizations:'):
                st.image('graphs/optimization_fitness.png')

                fig = analyzer.plot_history_graph(
                    portf_table[['Total Pos']]
                )
                st.pyplot(fig)

                fig = analyzer.plot_history_graph(
                    portf_table[solution.keys()]
                )
                st.pyplot(fig)

                fig = analyzer.plot_correlation_matrix(
                    portf_table[solution.keys()]
                )
                st.pyplot(fig)
                
                daily_return = portf_table.pct_change(1).dropna()
                fig = analyzer.plot_periodic_simple_returns(
                    daily_return[solution.keys()]
                )
                st.pyplot(fig)

                fig = analyzer.plot_PSR_risk(
                    daily_return[solution.keys()]
                )
                st.pyplot(fig)

if __name__ == '__main__':
    Dashboard.show()
