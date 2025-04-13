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
    
    def comprehensive_property_analysis(self):
        """
        Perform a comprehensive analysis of the Madrid property dataset
        
        Returns:
        --------
        dict : A dictionary containing various insights about the properties
        """
        # Property Price Analysis
        price_analysis = {
            'price_stats': {
                'min_price': self.df['PRICE'].min(),
                'max_price': self.df['PRICE'].max(),
                'mean_price': self.df['PRICE'].mean(),
                'median_price': self.df['PRICE'].median(),
            },
            'price_per_sqm_stats': {
                'min_price_per_sqm': self.df['UNITPRICE'].min(),
                'max_price_per_sqm': self.df['UNITPRICE'].max(),
                'mean_price_per_sqm': self.df['UNITPRICE'].mean(),
                'median_price_per_sqm': self.df['UNITPRICE'].median(),
            }
        }
        
        # Property Characteristics Analysis
        characteristics_analysis = {
            'room_distribution': self.df['ROOMNUMBER'].value_counts(normalize=True).to_dict(),
            'area_distribution': {
                'min_area': self.df['CONSTRUCTEDAREA'].min(),
                'max_area': self.df['CONSTRUCTEDAREA'].max(),
                'mean_area': self.df['CONSTRUCTEDAREA'].mean(),
                'median_area': self.df['CONSTRUCTEDAREA'].median(),
            },
            'amenities_prevalence': {
                'terrace_percentage': (self.df['HASTERRACE'] == 1).mean() * 100,
                'lift_percentage': (self.df['HASLIFT'] == 1).mean() * 100,
                'air_conditioning_percentage': (self.df['HASAIRCONDITIONING'] == 1).mean() * 100,
                'parking_space_percentage': (self.df['HASPARKINGSPACE'] == 1).mean() * 100,
                'box_room_percentage': (self.df['HASBOXROOM'] == 1).mean() * 100,
            }
        }
        
        # Location and Orientation Analysis
        location_analysis = {
            'orientation_distribution': {
                'north_facing': (self.df['HASNORTHORIENTATION'] == 1).mean() * 100,
                'south_facing': (self.df['HASSOUTHORIENTATION'] == 1).mean() * 100,
                'east_facing': (self.df['HASEASTORIENTATION'] == 1).mean() * 100,
                'west_facing': (self.df['HASWESTORIENTATION'] == 1).mean() * 100,
            },
            'distance_statistics': {
                'to_city_center': {
                    'min': self.df['DISTANCE_TO_CITY_CENTER'].min(),
                    'max': self.df['DISTANCE_TO_CITY_CENTER'].max(),
                    'mean': self.df['DISTANCE_TO_CITY_CENTER'].mean(),
                    'median': self.df['DISTANCE_TO_CITY_CENTER'].median(),
                },
                'to_metro': {
                    'min': self.df['DISTANCE_TO_METRO'].min(),
                    'max': self.df['DISTANCE_TO_METRO'].max(),
                    'mean': self.df['DISTANCE_TO_METRO'].mean(),
                    'median': self.df['DISTANCE_TO_METRO'].median(),
                }
            }
        }
        
        # Advanced Price Correlations
        price_correlations = {
            'price_correlations': {
                'price_vs_area': self.df['PRICE'].corr(self.df['CONSTRUCTEDAREA']),
                'price_vs_rooms': self.df['PRICE'].corr(self.df['ROOMNUMBER']),
                'price_vs_distance_to_center': self.df['PRICE'].corr(self.df['DISTANCE_TO_CITY_CENTER']),
            }
        }
        
        # Construction Year Analysis
        construction_analysis = {
            'construction_year_stats': {
                'min_year': self.df['CONSTRUCTIONYEAR'].min(),
                'max_year': self.df['CONSTRUCTIONYEAR'].max(),
                'mean_year': self.df['CONSTRUCTIONYEAR'].mean(),
                'median_year': self.df['CONSTRUCTIONYEAR'].median(),
            },
            'age_distribution': self.df['CONSTRUCTIONYEAR'].value_counts(bins=10, normalize=True)
        }
        
        return {
            'basic_info': self.basic_info,
            'price_analysis': price_analysis,
            'characteristics_analysis': characteristics_analysis,
            'location_analysis': location_analysis,
            'price_correlations': price_correlations,
            'construction_analysis': construction_analysis
        }
    
    def generate_advanced_visualizations(self, output_dir):
        """
        Generate advanced visualizations for the dataset
        
        Parameters:
        -----------
        output_dir : str, optional
            Directory to save the generated plots
        """
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # Price Distribution
        plt.figure(figsize=(12, 6))
        self.df['PRICE'].hist(bins=50)
        plt.title('Distribution of Property Prices in Madrid')
        plt.xlabel('Price (€)')
        plt.ylabel('Frequency')
        plt.savefig(f'{output_dir}/price_distribution.png')
        plt.close()
        
        # Price vs Area Scatter Plot
        plt.figure(figsize=(12, 6))
        plt.scatter(self.df['CONSTRUCTEDAREA'], self.df['PRICE'], alpha=0.1)
        plt.title('Property Price vs Constructed Area')
        plt.xlabel('Constructed Area (m²)')
        plt.ylabel('Price (€)')
        plt.savefig(f'{output_dir}/price_vs_area_scatter.png')
        plt.close()
        
        # Correlation Heatmap
        numeric_columns = ['PRICE', 'UNITPRICE', 'CONSTRUCTEDAREA', 'ROOMNUMBER', 
                           'BATHNUMBER', 'DISTANCE_TO_CITY_CENTER', 'CONSTRUCTIONYEAR']
        plt.figure(figsize=(12, 10))
        correlation_matrix = self.df[numeric_columns].corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
        plt.title('Correlation Heatmap of Numeric Property Features')
        plt.tight_layout()
        plt.savefig(f'{output_dir}/correlation_heatmap.png')
        plt.close()

    def plot_histogram(self, output_dir, features, bins=50, figsize=(20, 5), title=None):
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

        plt.show()
        plt.savefig(f'{output_dir}/Neat_Histogram.png')

    def printHead(self):
        """
        Print the first few rows of the dataset for quick inspection
        """
        print(self.df.head(1))



# Example usage
def main():
    file_path = (r"C:\Users\costa\Desktop\TFG\4.0 Estadística Básica para entender el data set\4.2 Funciones y cosas varias\Madrid_Sale.csv")
    output_dir = (r"4.0 Estadística Básica para entender el data set\4.2 Funciones y cosas varias\figures")

    analyzer = MadridPropertyAnalyzer(file_path)
    
    # Perform comprehensive analysis
    analysis_results = analyzer.comprehensive_property_analysis()
    
    # Print key insights
    print("--- Madrid Property Market Analysis ---")
    print("\nBasic Information:")
    print(f"Total Properties: {analysis_results['basic_info']['total_properties']}")
    print(f"Unique Properties: {analysis_results['basic_info']['total_unique_properties']}")
    
    print("\nPrice Analysis:")
    print(f"Mean Price: €{analysis_results['price_analysis']['price_stats']['mean_price']:,.2f}")
    print(f"Median Price: €{analysis_results['price_analysis']['price_stats']['median_price']:,.2f}")
    print(f"Mean Price per m²: €{analysis_results['price_analysis']['price_per_sqm_stats']['mean_price_per_sqm']:,.2f}")
    
    print("\nProperty Characteristics:")
    room_dist = analysis_results['characteristics_analysis']['room_distribution']
    print("Room Distribution:")
    for rooms, percentage in room_dist.items():
        print(f"{rooms} rooms: {percentage*100:.2f}%")
    
    print("\nAmenities Prevalence:")
    amenities = analysis_results['characteristics_analysis']['amenities_prevalence']
    for amenity, percentage in amenities.items():
        print(f"{amenity.replace('_', ' ').title()}: {percentage:.2f}%")


    # Generate histrograms for selected features
    print("\nGenerating histograms...")
    analyzer.plot_histogram(output_dir, features = ["PRICE", "CONSTRUCTEDAREA", "ROOMNUMBER", "BATHNUMBER", 
    "CONSTRUCTIONYEAR", "CADCONSTRUCTIONYEAR", "DISTANCE_TO_CITY_CENTER", 
    "DISTANCE_TO_METRO", "DISTANCE_TO_CASTELLANA", "LONGITUDE", "LATITUDE"], )

    # Generate advanced visualizations
    analyzer.generate_advanced_visualizations(output_dir)
    print(f"Advanced visualizations saved to: {output_dir}")
    

if __name__ == '__main__':
    main()