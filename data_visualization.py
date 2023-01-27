import pandas as pd
pd.plotting.register_matplotlib_converters()
import matplotlib.pyplot as plt
import seaborn as sns

def main():

    # Path of the file to read
    fifa_filepath = "data/fifa.csv"

    # Read the file into a variable fifa_data
    fifa_data = pd.read_csv(fifa_filepath, index_col="Date", parse_dates=True)

    # Print the first 5 rows of the data
    fifa_data.head()

    # Set the width and height of the figure
    plt.figure(figsize=(16,6))

    # Line chart showing how FIFA rankings evolved over time 
    sns.lineplot(data=fifa_data)

    plt.show()

    input()


if __name__ == "__main__":
    main()

