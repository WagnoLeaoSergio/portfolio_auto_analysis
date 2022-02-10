import numpy as np
import pandas as pd
import seaborn as sb
import yfinance as yf
from datetime import date
import matplotlib.pyplot as plt


class Portfolio_Analyzer():
    """
    Class responsible for the required components
    that generates the performance data and graphical
    visualization of the portfolio.
    """
    def __init__(self):
        self.graphs_folder = 'graphs/'
        pass

    def get_price_data(
        self,
        symbols: list,
        start_date: date,
        end_date: date
    ) -> pd.DataFrame:
        """
        Fetchs the historical data of the listed symbols.
        """
        print(f"Fetching {len(symbols)} assets data.")
        # Fetching data
        return yf.download(
            ' '.join(symbols),
            start=start_date.strftime('%Y-%m-%d'),
            end=end_date.strftime('%Y-%m-%d')
        )['Adj Close']

    def portfolio_table(
        self,
        data_frame: pd.DataFrame,
        percentages: list,
        allocation: float
    ) -> pd.DataFrame:

        """
        Creates an DataFrame with the normalized positions
        of each symbol associated with your respective percentage
        in the portfolio.
        """
        tables = [ pd.DataFrame(data_frame[i]) for i in data_frame.columns.values ]

        for i, value in enumerate(data_frame.columns.values):
            tables[i]['Norm return'] = tables[i] / tables[i].iloc[0]
            tables[i]['Allocation'] = tables[i]['Norm return'] * percentages[i]
            tables[i]['Position'] = tables[i]['Allocation'] * allocation


        all_tables = [ table['Position'] for table in tables ]
        portf_table = pd.concat(all_tables, axis=1)
        portf_table.columns = data_frame.columns.values
        portf_table['Total Pos'] = portf_table.sum(axis=1)
        return portf_table


    def plot_history_graph(self, data_frame: pd.DataFrame) -> None:
        """
        Plots the price history of the portfolio's symbols.
        """
        fig, ax = plt.subplots(figsize=(15, 8))
        for i in data_frame.columns.values:
            ax.plot(data_frame[i], label=i)

        ax.set_title("Portfolio Close Price History")
        ax.set_xlabel('Date', fontsize=18)
        ax.set_ylabel('Close Price INR (Rs)', fontsize=18)
        ax.legend(data_frame.columns.values, loc='upper left')
        plt.savefig(self.graphs_folder + 'symbols_prices_history.png')

    def plot_correlation_matrix(self, data_frame: pd.DataFrame) -> None:
        """
        Plot the correlation matrix of the portfolio.
        """
        correlation_matrix = data_frame.corr(method='pearson')
        print('Correlation between Symbols in your portfolio')
        print(correlation_matrix)

        plt.figure(figsize=(12, 7))
        sb.heatmap(
            correlation_matrix,
            xticklabels=correlation_matrix.columns,
            yticklabels=correlation_matrix.columns,
            cmap='YlGnBu',
            annot=True,
            linewidth=0.5
        )
        plt.savefig(self.graphs_folder + 'correlation_matrix.png')

    def periodic_simple_returns(
            self,
            data_frame: pd.DataFrame,
            period: int
    ) -> pd.DataFrame:
        """
        Creates a DataFrame with the period change in the
        symbol's price history.
        """
        return data_frame.pct_change(period).dropna()

    def plot_periodic_simple_returns(
            self,
            psr: pd.DataFrame
    ) -> None:
        """
        Plots the PSR (Periodic Simple Returns) graph.
        """
        fig, ax = plt.subplots(figsize=(15, 8))

        for i in psr.columns.values:
            ax.plot(psr[i], lw=2, label=i)

        ax.legend(loc='upper right', fontsize=10)
        ax.set_title('Volatility in periodic simple returns')
        ax.set_xlabel('Date')
        ax.set_ylabel('Periodic simple returns')

        plt.savefig(self.graphs_folder + 'periodic_simple_returns.png')

    def average_PSR(self, psr: pd.DataFrame) -> pd.DataFrame:
        """
        As the name says.
        """
        return psr.mean()

    def plot_PSR_risk(self, psr: pd.DataFrame) -> None:
        """
        Creates a BoxPlot visualization of the PSR.
        """
        print('Periodic Returns Risk')
        psr.plot(
            kind='box',
            figsize=(20, 10),
            title="Risk Box Plot"
        )
        plt.savefig(self.graphs_folder + 'risk_box_plot.png')

    def annualized_standard_deviation(
            self,
            psr: pd.DataFrame,
            days: int
    ):
        """
        Calculates annualization of the standard deviation
        for the PSR.
        """
        return (psr.std() * np.sqrt(days))

    def sharpe_ratio(
            self,
            avg_psr: pd.DataFrame,
            std: pd.DataFrame
    ) -> pd.DataFrame:
        return (avg_psr / std) * 100

    def f_sharpe_ratio(
        self,
        return_series: pd.DataFrame,
        N: int,
        risk_free: float
    ):
        mean = return_series.mean() * N  -risk_free
        sigma = return_series.std() * np.sqrt(N)
        return mean / sigma

    def f_sortino_ratio(
        self,
        return_series: pd.DataFrame,
        N: int,
        risk_free: float
    ):
        mean = return_series.mean() * N -risk_free
        std_neg = return_series[return_series < 0].std() * np.sqrt(N)
        return mean / std_neg

    def cummulative_PSR(self, psr: pd.DataFrame) -> pd.DataFrame:
        """
        Creates a DataFrame of the Cummulative PSR.
        """
        return (psr + 1).cumprod()

    def plot_cummulative_PSR(self, c_psr: pd.DataFrame) -> None:
        """
        Creates a Line plot for the Cummulative PSR.
        """
        fig, ax = plt.subplots(figsize=(18, 8))

        for i in c_psr.columns.values:
            ax.plot(
                    c_psr[i],
                    lw=2,
                    label=i
            )

        ax.legend(loc='upper left', fontsize=10)
        ax.set_title(
            'Periodic Cummulative Simple returns/growth of investment'
        )
        ax.set_xlabel('Date')
        ax.set_ylabel('Growth of Rs 1 investment')
        plt.savefig(self.graphs_folder + 'cummulative_returns.png')


def proto() -> None:
    plt.style.use('fivethirtyeight')

    # Defining parameters
    stocksymbols = [
        'TATAMOTORS',
        'DABUR',
        'ICICIBANK',
        'WIPRO',
        'BPCL',
        'IRCTC',
        'INFY',
        'RELIANCE'
    ]

    analyzer = Portfolio_Analyzer()

    start_date = date(2021, 1, 19)
    end_date = date.today()

    data_frame = analyzer.get_price_data(
        stocksymbols,
        start_date,
        end_date
    )

    print(data_frame.head())

    # Analysis

    # Price History
    analyzer.plot_history_graph(data_frame)

    # Correlation Matrix
    analyzer.plot_correlation_matrix(data_frame)

    # Daily Simple Returns
    print('Daily Simple Returns:')
    daily_simple_return = analyzer.periodic_simple_returns(
            data_frame,
            1
    )
    print(daily_simple_return.head())

    analyzer.plot_periodic_simple_returns(daily_simple_return)

    # Avrage Daily returns
    print('Average Daily returns(%) of stocks in your portfolio')
    avg_daily = analyzer.average_PSR(daily_simple_return)
    print(avg_daily*100)

    # Risk
    analyzer.plot_PSR_risk(daily_simple_return)

    standard_deviation = analyzer.annualized_standard_deviation(
        daily_simple_return,
        252
    )

    print('Sharpe Ratio')
    return_per_unit_risk = analyzer.sharpe_ratio(
            avg_daily,
            standard_deviation
    )
    print(return_per_unit_risk)

    # Cumulative returns
    print('Cummulative Returns:')
    daily_cummulative_simple_return = analyzer.cummulative_PSR(
            daily_simple_return
    )
    print(daily_cummulative_simple_return)
    analyzer.plot_cummulative_PSR(daily_cummulative_simple_return)
