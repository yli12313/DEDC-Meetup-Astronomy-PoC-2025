import psycopg2
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
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

def create_advanced_plots(df):
    """Generate advanced statistical plots and visualizations"""
    
    # Calculate derived metrics
    df['density_ratio'] = df['pl_masse'] / (df['pl_rade'] ** 3)
    df['surface_gravity'] = df['pl_masse'] / (df['pl_rade'] ** 2)  # Relative to Earth
    df['escape_velocity'] = np.sqrt(2 * df['pl_masse'] / df['pl_rade'])  # Relative to Earth
    
    # Planet classification
    def classify_planet(radius, mass):
        if radius < 1.25:
            return 'Earth-like'
        elif radius < 2.0:
            return 'Super-Earth'
        elif radius < 4.0:
            if mass < 10:
                return 'Mini-Neptune'
            else:
                return 'Neptune-like'
        else:
            return 'Gas Giant'
    
    df['planet_type'] = df.apply(lambda x: classify_planet(x['pl_rade'], x['pl_masse']), axis=1)
    
    # Set up matplotlib style
    plt.style.use('dark_background')
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD', '#98D8C8']
    
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 24))
    
    # Plot 1: Advanced Mass-Radius Relationship with Theoretical Lines
    plt.subplot(4, 3, 1)
    
    # Theoretical composition lines
    radii = np.linspace(0.5, 2.0, 100)
    iron_masses = radii ** 3 * 5.5  # Iron planet
    rock_masses = radii ** 3 * 1.0  # Earth-like
    water_masses = radii ** 3 * 0.3  # Water world
    
    plt.plot(radii, iron_masses, '--', color='red', alpha=0.7, label='Pure Iron', linewidth=2)
    plt.plot(radii, rock_masses, '--', color='brown', alpha=0.7, label='Earth-like Rock', linewidth=2)
    plt.plot(radii, water_masses, '--', color='blue', alpha=0.7, label='Water World', linewidth=2)
    
    # Actual data points
    for i, planet_type in enumerate(df['planet_type'].unique()):
        subset = df[df['planet_type'] == planet_type]
        plt.scatter(subset['pl_rade'], subset['pl_masse'], 
                   label=planet_type, alpha=0.8, s=60, color=colors[i])
    
    plt.xlabel('Planet Radius (Earth Radii)', fontsize=12, fontweight='bold')
    plt.ylabel('Planet Mass (Earth Masses)', fontsize=12, fontweight='bold')
    plt.title('Mass-Radius Diagram with Composition Models', fontsize=14, fontweight='bold')
    plt.legend(loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.xlim(0.4, 2.2)
    plt.ylim(0, 15)
    
    # Plot 2: Density Distribution with Statistical Analysis
    plt.subplot(4, 3, 2)
    
    log_density = np.log10(df['density_ratio'])
    
    # Create histogram
    n, bins, patches = plt.hist(log_density, bins=25, alpha=0.7, color='#4ECDC4', edgecolor='black')
    
    # Fit normal distribution
    mu, std = stats.norm.fit(log_density)
    x = np.linspace(log_density.min(), log_density.max(), 100)
    p = stats.norm.pdf(x, mu, std)
    plt.plot(x, p * len(log_density) * (bins[1] - bins[0]), 
             'r-', linewidth=3, label=f'Normal Fit (μ={mu:.2f}, σ={std:.2f})')
    
    plt.axvline(0, color='yellow', linestyle='--', linewidth=2, label='Earth Density')
    plt.xlabel('Log₁₀(Density Ratio)', fontsize=12, fontweight='bold')
    plt.ylabel('Frequency', fontsize=12, fontweight='bold')
    plt.title('Planet Density Distribution Analysis', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 3: Machine Learning Clustering
    plt.subplot(4, 3, 3)
    
    # Prepare data for clustering
    features = df[['pl_rade', 'pl_masse', 'density_ratio', 'surface_gravity']].values
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    # Perform K-means clustering
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(features_scaled)
    
    scatter = plt.scatter(df['pl_rade'], df['pl_masse'], c=clusters, 
                         cmap='viridis', s=80, alpha=0.8)
    plt.colorbar(scatter, label='Cluster')
    plt.xlabel('Planet Radius (Earth Radii)', fontsize=12, fontweight='bold')
    plt.ylabel('Planet Mass (Earth Masses)', fontsize=12, fontweight='bold')
    plt.title('ML Clustering of Exoplanets', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # Plot 4: Surface Gravity vs Escape Velocity
    plt.subplot(4, 3, 4)
    
    for i, planet_type in enumerate(df['planet_type'].unique()):
        subset = df[df['planet_type'] == planet_type]
        plt.scatter(subset['surface_gravity'], subset['escape_velocity'], 
                   label=planet_type, alpha=0.8, s=60, color=colors[i])
    
    # Add Earth reference
    plt.scatter(1.0, 1.0, color='white', s=200, marker='*', 
               edgecolors='red', linewidth=2, label='Earth', zorder=10)
    
    plt.xlabel('Surface Gravity (Earth = 1)', fontsize=12, fontweight='bold')
    plt.ylabel('Escape Velocity (Earth = 1)', fontsize=12, fontweight='bold')
    plt.title('Planetary Environment Analysis', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.loglog()
    
    # Plot 5: Host Star System Analysis
    plt.subplot(4, 3, 5)
    
    # Calculate system statistics
    system_stats = df.groupby('hostname').agg({
        'pl_name': 'count',
        'pl_masse': ['mean', 'std'],
        'pl_rade': ['mean', 'std']
    }).round(2)
    
    system_stats.columns = ['planet_count', 'mass_mean', 'mass_std', 'radius_mean', 'radius_std']
    system_stats = system_stats.reset_index()
    
    # Plot systems with multiple planets
    multi_planet = system_stats[system_stats['planet_count'] > 1]
    
    bubble_sizes = multi_planet['planet_count'] * 50
    scatter = plt.scatter(multi_planet['radius_mean'], multi_planet['mass_mean'], 
                         s=bubble_sizes, alpha=0.6, c=multi_planet['planet_count'], 
                         cmap='plasma', edgecolors='white', linewidth=1)
    
    plt.colorbar(scatter, label='Number of Planets')
    plt.xlabel('Mean Planet Radius (Earth Radii)', fontsize=12, fontweight='bold')
    plt.ylabel('Mean Planet Mass (Earth Masses)', fontsize=12, fontweight='bold')
    plt.title('Multi-Planet System Characteristics', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # Plot 6: Statistical Correlation Matrix
    plt.subplot(4, 3, 6)
    
    correlation_data = df[['pl_rade', 'pl_masse', 'density_ratio', 'surface_gravity', 'escape_velocity']]
    correlation_matrix = correlation_data.corr()
    
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
    sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='coolwarm', 
                center=0, square=True, fmt='.2f', cbar_kws={'label': 'Correlation'})
    plt.title('Parameter Correlation Matrix', fontsize=14, fontweight='bold')
    
    # Plot 7: Habitability Zone Analysis (simplified)
    plt.subplot(4, 3, 7)
    
    # Simple habitability scoring based on size and mass
    def habitability_score(radius, mass):
        size_score = max(0, 1 - abs(radius - 1.0) / 2.0)  # Closer to Earth size = better
        mass_score = max(0, 1 - abs(mass - 1.0) / 10.0)   # Closer to Earth mass = better
        return (size_score + mass_score) / 2
    
    df['habitability'] = df.apply(lambda x: habitability_score(x['pl_rade'], x['pl_masse']), axis=1)
    
    # Plot habitability vs planet properties
    scatter = plt.scatter(df['pl_rade'], df['pl_masse'], c=df['habitability'], 
                         cmap='RdYlGn', s=80, alpha=0.8, edgecolors='black', linewidth=0.5)
    
    plt.colorbar(scatter, label='Habitability Score')
    plt.scatter(1.0, 1.0, color='blue', s=200, marker='*', 
               edgecolors='white', linewidth=2, label='Earth', zorder=10)
    
    plt.xlabel('Planet Radius (Earth Radii)', fontsize=12, fontweight='bold')
    plt.ylabel('Planet Mass (Earth Masses)', fontsize=12, fontweight='bold')
    plt.title('Exoplanet Habitability Assessment', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 8: Discovery Timeline Analysis
    plt.subplot(4, 3, 8)
    
    df['rowupdate'] = pd.to_datetime(df['rowupdate'])
    df['discovery_year'] = df['rowupdate'].dt.year
    
    yearly_discoveries = df['discovery_year'].value_counts().sort_index()
    cumulative_discoveries = yearly_discoveries.cumsum()
    
    plt.plot(yearly_discoveries.index, cumulative_discoveries.values, 
             marker='o', linewidth=3, markersize=8, color='#FF6B6B')
    plt.fill_between(yearly_discoveries.index, cumulative_discoveries.values, 
                     alpha=0.3, color='#FF6B6B')
    
    plt.xlabel('Year', fontsize=12, fontweight='bold')
    plt.ylabel('Cumulative TESS Discoveries', fontsize=12, fontweight='bold')
    plt.title('TESS Exoplanet Discovery Timeline', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # Plot 9: Size Distribution by Planet Type
    plt.subplot(4, 3, 9)
    
    planet_types = df['planet_type'].unique()
    for i, ptype in enumerate(planet_types):
        subset = df[df['planet_type'] == ptype]
        plt.hist(subset['pl_rade'], alpha=0.7, label=ptype, 
                color=colors[i], bins=15, density=True)
    
    plt.xlabel('Planet Radius (Earth Radii)', fontsize=12, fontweight='bold')
    plt.ylabel('Probability Density', fontsize=12, fontweight='bold')
    plt.title('Radius Distribution by Planet Type', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 10: 3D Visualization
    ax = plt.subplot(4, 3, 10, projection='3d')
    
    scatter = ax.scatter(df['pl_rade'], df['pl_masse'], df['density_ratio'], 
                        c=df['habitability'], cmap='viridis', s=60, alpha=0.8)
    
    ax.set_xlabel('Radius (Earth Radii)', fontweight='bold')
    ax.set_ylabel('Mass (Earth Masses)', fontweight='bold')
    ax.set_zlabel('Density Ratio', fontweight='bold')
    ax.set_title('3D Exoplanet Parameter Space', fontweight='bold')
    
    # Plot 11: Statistical Distribution Comparison
    plt.subplot(4, 3, 11)
    
    # Compare mass distributions of different planet types
    planet_masses = [df[df['planet_type'] == ptype]['pl_masse'].values 
                     for ptype in planet_types[:4]]  # Limit to 4 types
    
    plt.boxplot(planet_masses, labels=planet_types[:4])
    plt.ylabel('Planet Mass (Earth Masses)', fontsize=12, fontweight='bold')
    plt.title('Mass Distribution by Planet Type', fontsize=14, fontweight='bold')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    # Plot 12: Advanced Statistics Panel
    plt.subplot(4, 3, 12)
    
    # Create summary statistics visualization
    stats_text = f"""
ADVANCED STATISTICS SUMMARY

Total Planets: {len(df)}
Unique Host Systems: {df['hostname'].nunique()}

MASS-RADIUS RELATIONSHIP:
Correlation: r = {df['pl_masse'].corr(df['pl_rade']):.3f}
Power Law Fit: M ∝ R^{np.polyfit(np.log(df['pl_rade']), np.log(df['pl_masse']), 1)[0]:.2f}

DENSITY ANALYSIS:
Mean Density Ratio: {df['density_ratio'].mean():.2f} ± {df['density_ratio'].std():.2f}
Skewness: {stats.skew(df['density_ratio']):.2f}
Kurtosis: {stats.kurtosis(df['density_ratio']):.2f}

HABITABILITY METRICS:
High Habitability (>0.7): {len(df[df['habitability'] > 0.7])} planets
Earth-like Size (0.8-1.2 R⊕): {len(df[(df['pl_rade'] > 0.8) & (df['pl_rade'] < 1.2)])} planets
Super-Earth Size (1.2-2.0 R⊕): {len(df[(df['pl_rade'] > 1.2) & (df['pl_rade'] < 2.0)])} planets

DISCOVERY TRENDS:
Most Recent Discovery: {df['discovery_year'].max()}
Peak Discovery Year: {yearly_discoveries.idxmax()} ({yearly_discoveries.max()} planets)
    """
    
    plt.text(0.05, 0.95, stats_text, transform=plt.gca().transAxes, 
             fontsize=10, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='black', alpha=0.8))
    plt.axis('off')
    
    plt.tight_layout()
    
    # Save the plot
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"advanced_exoplanet_analysis_{timestamp}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='black')
    print(f"Advanced analysis plot saved as {filename}")
    
    plt.show()
    
    return df

def create_interactive_plots(df):
    """Create interactive Plotly visualizations"""
    
    print("Creating interactive visualizations...")
    
    # Interactive 3D scatter plot
    fig_3d = go.Figure(data=[go.Scatter3d(
        x=df['pl_rade'],
        y=df['pl_masse'],
        z=df['density_ratio'],
        mode='markers',
        marker=dict(
            size=8,
            color=df['habitability'],
            colorscale='Viridis',
            colorbar=dict(title="Habitability Score"),
            opacity=0.8
        ),
        text=df['pl_name'],
        hovertemplate='<b>%{text}</b><br>' +
                      'Radius: %{x:.2f} R⊕<br>' +
                      'Mass: %{y:.2f} M⊕<br>' +
                      'Density: %{z:.2f}<br>' +
                      '<extra></extra>'
    )])
    
    fig_3d.update_layout(
        title='Interactive 3D Exoplanet Analysis',
        scene=dict(
            xaxis_title='Planet Radius (Earth Radii)',
            yaxis_title='Planet Mass (Earth Masses)',
            zaxis_title='Density Ratio',
            bgcolor='black'
        ),
        paper_bgcolor='black',
        font=dict(color='white')
    )
    
    # Save interactive plot
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    interactive_filename = f"interactive_exoplanet_3d_{timestamp}.html"
    fig_3d.write_html(interactive_filename)
    print(f"Interactive 3D plot saved as {interactive_filename}")

def main():
    """Main function for advanced analytics"""
    print("Starting advanced exoplanet analytics...")
    
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
        
        # Create advanced plots
        df = create_advanced_plots(df)
        
        # Create interactive visualizations
        create_interactive_plots(df)
        
        print("\nAdvanced analytics completed successfully!")
        
    finally:
        conn.close()

if __name__ == "__main__":
    main()