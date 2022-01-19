import argparse  # pragma: no cover

from . import BaseClass, base_function  # pragma: no cover


def main() -> None:  # pragma: no cover
    """
    The main function executes on commands:
    `python -m portfolio_auto_analysis` and `$ portfolio_auto_analysis `.

    This is your program's entry point.

    You can change this function to do whatever you want.
    Examples:
        * Run a test suite
        * Run a server
        * Do some other stuff
        * Run a command line application (Click, Typer, ArgParse)
        * List all available tasks
        * Run an application (Flask, FastAPI, Django, etc.)
    """
    parser = argparse.ArgumentParser(
        description="portfolio_auto_analysis.",
        epilog="Enjoy the portfolio_auto_analysis functionality!",
    )
    # This is required positional argument
    parser.add_argument(
        "name",
        type=str,
        help="The username",
        default="WagnoLeaoSergio",
    )
    # This is optional named argument
    parser.add_argument(
        "-m",
        "--message",
        type=str,
        help="The Message",
        default="Hello",
        required=False,
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Optionally adds verbosity",
    )
    args = parser.parse_args()
    print(f"{args.message} {args.name}!")
    if args.verbose:
        print("Verbose mode is on.")

    print("Executing main function")
    # base = BaseClass()
    # print(base.base_method())
    # print(base_function())
    proto()
    print("End of main function")

def proto() -> None:
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sb
    from datetime import date
    from nsepy import get_history
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

    start_date = date(2021,1,19)
    end_date = date.today()
    print(end_date)
    print(f"You have {len(stocksymbols)} assets in your portfolio.")

    # Fetching data
    data_frame = pd.DataFrame()
    for i, symbol in enumerate(stocksymbols):
        data = get_history(
            symbol=symbol,
            start=start_date,
            end=end_date
        )[['Symbol', 'Close']]
        
        data.rename(
            columns={ 'Close': data['Symbol'][0] },
            inplace=True
        )

        data.drop(['Symbol'], axis=1, inplace=True)

        if i == 0:
            data_frame = data
        else:
            data_frame = data_frame.join(data)

    print(data_frame.head())


    # Analysis
    # Price History
    graphs_folder = 'graphs/'

    fig, ax = plt.subplots(figsize=(15,8))
    for i in data_frame.columns.values:
        ax.plot(data_frame[i], label=i)

    ax.set_title("Portfolio Close Price History")
    ax.set_xlabel('Date', fontsize=18)
    ax.set_ylabel('Close Price INR (Rs)', fontsize=18)
    ax.legend(data_frame.columns.values, loc='upper left')
    plt.savefig(graphs_folder + 'portfolio_prices.png')

    # Correlation Matrix
    correlation_matrix = data_frame.corr(method='pearson')
    print('Correlation between Stocks in your portfolio')
    print(correlation_matrix)

    fig = plt.figure(figsize=(12,7))
    sb.heatmap(
        correlation_matrix,
        xticklabels=correlation_matrix.columns,
        yticklabels=correlation_matrix.columns,
        cmap='YlGnBu',
        annot=True,
        linewidth=0.5
    )
    plt.savefig(graphs_folder + 'correlation_matrix.png')

    # Daily Simple Returns
    print('Daily Simple Returns:')
    daily_simple_return = data_frame.pct_change(1)
    daily_simple_return.dropna(inplace=True)
    print(daily_simple_return.head())

    fig, ax = plt.subplots(figsize=(15, 8))

    for i in daily_simple_return.columns.values:
        ax.plot(daily_simple_return[i], lw=2, label=i)

    ax.legend(loc='upper right', fontsize=10)
    ax.set_title('Volatility in Daily simple returns')
    ax.set_xlabel('Date')
    ax.set_ylabel('Daily simple returns')

    plt.savefig(graphs_folder + 'daily_simple_returns.png')

    # Avrage Daily returns
    print('Average Daily returns(%) of stocks in your portfolio')
    avg_daily = daily_simple_return.mean()
    print(avg_daily*100)

    # Risk
    print('Daily Returns Risk')
    daily_simple_return.plot(
        kind='box',
        figsize=(20,10),
        title="Risk Box Plot"
    )
    plt.savefig(graphs_folder + 'risk_box_plot.png')

    print('Annualized Standard Deviation (Volatility(%), 252 trading days) ' +
        'of individual stocks in your portfolio on the basis of daily simple ' +
        'returns.')

    standard_deviation = (daily_simple_return.std() * np.sqrt(252))
    print(standard_deviation)

    return_per_unit_risk = (avg_daily / standard_deviation) * 100

    print('Sharpe Ratio')
    print(return_per_unit_risk)
    
    # Cumulative returns
    print('Cummulative Returns:')
    daily_cummulative_simple_return = (daily_simple_return + 1).cumprod()
    print(daily_cummulative_simple_return)

    fig, ax = plt.subplots(figsize=(18,8))
    
    for i in daily_cummulative_simple_return.columns.values:
        ax.plot(
                daily_cummulative_simple_return[i],
                lw=2,
                label=i
        )

    ax.legend(loc='upper left', fontsize=10)
    ax.set_title('Daily Cummulative Simple returns/growth of investment')
    ax.set_xlabel('Date')
    ax.set_ylabel('Growth of Rs 1 investment')

    plt.savefig(graphs_folder + 'cummulative_returns.png')

if __name__ == "__main__":  # pragma: no cover
    main()
