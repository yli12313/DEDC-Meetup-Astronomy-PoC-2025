import psycopg2
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from datetime import datetime

def connect_to_database():
    """Connect to PostgreSQL database"""
    try:
        conn = psycopg2.connect(
            host="localhost",
            database="meetup_demo",
            user=os.getenv("DB_USER", "postgres"),
            password=os.getenv("DB_PASSWORD", ""),
            port=os.getenv("DB_PORT", "5432")
        )
        return conn
    except psycopg2.Error as e:
        print(f"Error connecting to database: {e}")
        return None

def load_data_from_database(conn):
    """Load exoplanet data from database into pandas DataFrame"""
    query = """
    SELECT hostname, pl_name, pl_rade, pl_masse, rowupdate
    FROM exoplanets
    WHERE pl_rade IS NOT NULL AND pl_masse IS NOT NULL
    ORDER BY hostname, pl_name;
    """
    
    try:
        df = pd.read_sql_query(query, conn)
        print(f"Loaded {len(df)} exoplanet records from database")
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def create_plots(df):
    """Generate interesting plots about the exoplanets"""
    
    # Set up the plotting style
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")
    
    # Create figure with subplots
    fig = plt.figure(figsize=(16, 12))
    
    # Plot 1: Mass vs Radius scatter plot
    plt.subplot(2, 3, 1)
    plt.scatter(df['pl_rade'], df['pl_masse'], alpha=0.6, s=50)
    plt.xlabel('Planet Radius (Earth Radii)')
    plt.ylabel('Planet Mass (Earth Masses)')
    plt.title('Exoplanet Mass vs Radius')
    plt.grid(True, alpha=0.3)
    
    # Add Earth reference point
    plt.scatter(1.0, 1.0, color='red', s=100, marker='*', label='Earth', zorder=5)
    plt.legend()
    
    # Plot 2: Planet radius distribution
    plt.subplot(2, 3, 2)
    plt.hist(df['pl_rade'], bins=20, alpha=0.7, edgecolor='black')
    plt.axvline(df['pl_rade'].mean(), color='red', linestyle='--', label=f'Mean: {df["pl_rade"].mean():.2f}')
    plt.axvline(1.0, color='orange', linestyle='--', label='Earth Radius')
    plt.xlabel('Planet Radius (Earth Radii)')
    plt.ylabel('Number of Planets')
    plt.title('Distribution of Planet Radii')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 3: Planet mass distribution
    plt.subplot(2, 3, 3)
    plt.hist(df['pl_masse'], bins=20, alpha=0.7, edgecolor='black')
    plt.axvline(df['pl_masse'].mean(), color='red', linestyle='--', label=f'Mean: {df["pl_masse"].mean():.2f}')
    plt.axvline(1.0, color='orange', linestyle='--', label='Earth Mass')
    plt.xlabel('Planet Mass (Earth Masses)')
    plt.ylabel('Number of Planets')
    plt.title('Distribution of Planet Masses')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 4: Density calculation and plot
    plt.subplot(2, 3, 4)
    # Calculate density (mass/volume ratio, normalized to Earth)
    df['density_ratio'] = df['pl_masse'] / (df['pl_rade'] ** 3)
    earth_density_ratio = 1.0  # Earth reference
    
    plt.scatter(df['pl_rade'], df['density_ratio'], alpha=0.6, s=50)
    plt.axhline(earth_density_ratio, color='red', linestyle='--', label='Earth Density')
    plt.xlabel('Planet Radius (Earth Radii)')
    plt.ylabel('Density Ratio (relative to Earth)')
    plt.title('Planet Density vs Radius')
    plt.yscale('log')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 5: Top 10 most prolific host systems
    plt.subplot(2, 3, 5)
    host_counts = df['hostname'].value_counts().head(10)
    host_counts.plot(kind='bar')
    plt.xlabel('Host Star')
    plt.ylabel('Number of Planets')
    plt.title('Top 10 Most Prolific Host Systems')
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, alpha=0.3)
    
    # Plot 6: Mass vs Radius with planet classification
    plt.subplot(2, 3, 6)
    
    # Define planet types based on radius
    def classify_planet(radius):
        if radius < 1.25:
            return 'Earth-like'
        elif radius < 2.0:
            return 'Super-Earth'
        elif radius < 4.0:
            return 'Mini-Neptune'
        else:
            return 'Neptune-like'
    
    df['planet_type'] = df['pl_rade'].apply(classify_planet)
    
    # Plot each type with different colors
    for planet_type in df['planet_type'].unique():
        subset = df[df['planet_type'] == planet_type]
        plt.scatter(subset['pl_rade'], subset['pl_masse'], 
                   label=planet_type, alpha=0.7, s=50)
    
    plt.xlabel('Planet Radius (Earth Radii)')
    plt.ylabel('Planet Mass (Earth Masses)')
    plt.title('Exoplanet Classification')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the plot
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"exoplanet_analysis_{timestamp}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Plot saved as {filename}")
    
    plt.show()
    
    return df

def print_statistics(df):
    """Print interesting statistics about the dataset"""
    print("\n" + "="*50)
    print("EXOPLANET STATISTICS")
    print("="*50)
    
    print(f"Total planets: {len(df)}")
    print(f"Unique host systems: {df['hostname'].nunique()}")
    print(f"Average planets per system: {len(df) / df['hostname'].nunique():.2f}")
    
    print(f"\nRADIUS STATISTICS:")
    print(f"Mean radius: {df['pl_rade'].mean():.2f} Earth radii")
    print(f"Median radius: {df['pl_rade'].median():.2f} Earth radii")
    print(f"Range: {df['pl_rade'].min():.2f} - {df['pl_rade'].max():.2f} Earth radii")
    
    print(f"\nMASS STATISTICS:")
    print(f"Mean mass: {df['pl_masse'].mean():.2f} Earth masses")
    print(f"Median mass: {df['pl_masse'].median():.2f} Earth masses")
    print(f"Range: {df['pl_masse'].min():.2f} - {df['pl_masse'].max():.2f} Earth masses")
    
    # Planet classification counts
    print(f"\nPLANET CLASSIFICATION:")
    for planet_type, count in df['planet_type'].value_counts().items():
        percentage = (count / len(df)) * 100
        print(f"{planet_type}: {count} ({percentage:.1f}%)")
    
    # Most prolific systems
    print(f"\nTOP 5 HOST SYSTEMS:")
    for host, count in df['hostname'].value_counts().head(5).items():
        print(f"{host}: {count} planets")

def main():
    """Main function to analyze exoplanet data"""
    print("Starting exoplanet analysis...")
    
    # Connect to database
    conn = connect_to_database()
    if not conn:
        return
    
    try:
        # Load data
        df = load_data_from_database(conn)
        if df is None or df.empty:
            print("No data found in database")
            return
        
        # Create plots
        df = create_plots(df)
        
        # Print statistics
        print_statistics(df)
        
    finally:
        conn.close()

if __name__ == "__main__":
    main()