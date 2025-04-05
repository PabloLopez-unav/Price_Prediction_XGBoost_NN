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

    def plot_histogram(self, feature, bins=30, figsize=(10, 6), title=None):
        """
        Plot histogram of a specific feature in the dataset
        
        Parameters:
        -----------
        feature : str
            Column name of the feature to plot
        bins : int, default=30
            Number of bins for the histogram
        figsize : tuple, default=(10, 6)
            Figure size (width, height) in inches
        title : str, optional
            Custom title for the plot. If None, a default title will be used
        """
        if feature not in self.df.columns:
            print(f"Feature '{feature}' not found in the dataset.")
            print(f"Available features: {', '.join(self.df.columns)}")
            return
        
        plt.figure(figsize=figsize)
        plt.hist(self.df[feature].dropna(), bins=bins, alpha=0.7, color='skyblue', edgecolor='black')
        
        # Set plot title
        if title is None:
            title = f'Histogram of {feature}'
        plt.title(title, fontsize=14)
        
        plt.xlabel(feature, fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()

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

    # After all printing is done, execute the visualization functions
    print("\nGenerating visualizations...")

    # Get all numeric columns for histograms
    numeric_cols = analyzer.df.select_dtypes(include=[np.number]).columns.tolist()

    # Calculate number of rows needed (4 columns)
    n_rows = (len(numeric_cols) + 3) // 4  # Redondeo hacia arriba

    # Create figure with grid of subplots (4 columns)
    fig, axes = plt.subplots(nrows=n_rows, ncols=4, figsize=(20, 5*n_rows))
    fig.tight_layout(pad=3.0)

    # Plot histogram for each numeric feature
    for i, col in enumerate(numeric_cols):
        row_idx = i // 4
        col_idx = i % 4
        analyzer.df[col].hist(bins=50, 
                            ax=axes[row_idx, col_idx],  # Acceso correcto al subplot
                            alpha=0.7, 
                            color='skyblue', 
                            edgecolor='black')
        axes[row_idx, col_idx].set_title(f'Distribution of {col}', fontsize=10)
        axes[row_idx, col_idx].set_xlabel(col, fontsize=8)
        axes[row_idx, col_idx].grid(alpha=0.3)

    # Hide empty subplots if any
    for j in range(len(numeric_cols), n_rows*4):
        row_idx = j // 4
        col_idx = j % 4
        axes[row_idx, col_idx].axis('off')

    plt.show()
    plt.savefig(f'{output_dir}/historigrama_de_todo.png')


    # Generate advanced visualizations
    analyzer.generate_advanced_visualizations(output_dir)
    print(f"Advanced visualizations saved to: {output_dir}")
    

if __name__ == '__main__':
    main()