import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class MadridPropertyAnalyzer:
    def __init__(self, file_path):
        """
        Initialize the analyzer with the Madrid property sales dataset
        
        Parameters:
        -----------
        file_path : str
            Path to the CSV file containing Madrid property sales data
        """
        # Read the CSV file 

        self.df = pd.read_csv(file_path, 
                               low_memory=False,  # Handle mixed data types
                               parse_dates=False)  # Avoid date parsing overhead
        
        # Basic dataset information
        self.basic_info = {
            'total_properties': len(self.df),
            'total_unique_properties': self.df['ASSETID'].nunique(),
            'columns': self.df.columns.tolist()
        }
    

    def plot_histogram(self, features, bins=85, figsize=(20, 5), title=None):
        """
        features : list of str
            List of column names to plot.
        bins : int, default=50
            Number of bins for each histogram.
        figsize : tuple, default=(20, 5)
            Base size for each row of histograms (width, height).
        title : str, optional
            Title for the full figure. If None, no main title is set.
        """
        import matplotlib.pyplot as plt
        import numpy as np

        print("\nGenerating visualizations...")

        # Check if features exist in the DataFrame
        missing = [col for col in features if col not in self.df.columns]
        if missing:
            print(f"Warning: These features were not found in the dataset: {', '.join(missing)}")
            features = [col for col in features if col in self.df.columns]

        if not features:
            print("No valid features provided for plotting.")
            return

        # Calculate grid size
        n_cols = 4
        n_rows = (len(features) + n_cols - 1) // n_cols

        fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(figsize[0], figsize[1]*n_rows))
        fig.tight_layout(pad=3.0)
        axes = axes.flatten()

        # Plot each selected feature
        for i, col in enumerate(features):
            self.df[col].dropna().hist(bins=bins, 
                                    ax=axes[i], 
                                    alpha=0.7, 
                                    color='skyblue', 
                                    edgecolor='black')
            axes[i].set_title(f'Distribution of {col}', fontsize=10)
            axes[i].set_xlabel(col, fontsize=8)
            axes[i].grid(alpha=0.3)

        # Turn off unused subplots
        for j in range(len(features), len(axes)):
            axes[j].axis('off')

        if title:
            fig.suptitle(title, fontsize=16)
            plt.subplots_adjust(top=0.95)

        plt.show(block=False)


    def filter_tails(self, features, percent=0.9):
        """
        Parameters:
        -----------
        features : list of str
            Columns to filter.
        percent : float, default=0.9
            Percentage of central data to keep (between 0 and 1).
        -----------
        """
        df_filtered = self.df.copy()
        tail = (1 - percent) / 2  # For 90%, tail = 0.05

        for col in features:
            if col not in df_filtered.columns:
                print(f"Warning: Column '{col}' not found.")
                continue
            if not np.issubdtype(df_filtered[col].dtype, np.number):
                print(f"Warning: Column '{col}' is not numeric.")
                continue

            lower = df_filtered[col].quantile(tail)
            upper = df_filtered[col].quantile(1 - tail)
            df_filtered = df_filtered[(df_filtered[col] >= lower) & (df_filtered[col] <= upper)]

        # Create a new instance of MadridPropertyAnalyzer with the filtered data
        new_analyzer = MadridPropertyAnalyzer.__new__(MadridPropertyAnalyzer)  # Create an instance manually
        new_analyzer.df = df_filtered  # Assign the filtered DataFrame
        new_analyzer.basic_info = {
            'total_properties': len(df_filtered),
            'total_unique_properties': df_filtered['ASSETID'].nunique(),
            'columns': df_filtered.columns.tolist()
        }
        
        return new_analyzer



# Example usage
def main():
    file_path = (r"C:\Users\costa\Desktop\TFG\4.0 Estadística Básica para entender el data set\4.2 Funciones y cosas varias\Madrid_Sale.csv")
    output_dir = (r"4.0 Estadística Básica para entender el data set\4.2 Funciones y cosas varias\figures")
    features = ["PRICE", "CONSTRUCTEDAREA", "ROOMNUMBER", "BATHNUMBER", "CONSTRUCTIONYEAR", "CADCONSTRUCTIONYEAR", "DISTANCE_TO_CITY_CENTER", 
    "DISTANCE_TO_METRO", "DISTANCE_TO_CASTELLANA", "LONGITUDE", "LATITUDE"]

    analyzer = MadridPropertyAnalyzer(file_path)

    analyzer_NO_TAILS = analyzer.filter_tails(features, percent=0.975)
    
    # Print key insights
    print("--- Madrid Property Market Analysis ---")

    # Generate histrograms for selected features
    print("\nGenerating histograms...")
    analyzer.plot_histogram(features)
    plt.savefig(f'{output_dir}/Neat_Histogram.png')
    analyzer_NO_TAILS.plot_histogram(features, title="Filtered Histograms (No Tails)")
    plt.savefig(f'{output_dir}/Neat_Histogram_NO_TAILS.png')

if __name__ == '__main__':
    main()