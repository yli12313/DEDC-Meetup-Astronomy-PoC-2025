import requests
from datetime import datetime

def fetch_exoplanet_data():
    """
    """

    """Fetch exoplanet data from NASA Exoplanet Archive"""
    api = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync?query=select+hostname,pl_name,pl_rade,pl_masse+from+ps+where+pl_rade+<+=+1.8+and+pl_masse+>+0+and+disc_facility+=+'Transiting Exoplanet Survey Satellite (TESS)'&format=csv"
    
    try:
        print("Fetching exoplanet data...")
        response = requests.get(api, timeout=30)
        response.raise_for_status()
        
        return response.text
    
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data: {e}")
        return None

def save_csv_data(data, filename=None):
    """Save CSV data to file"""
    if not data:
        print("No data to save")
        return None
    
    if not filename:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"tess_exoplanets_{timestamp}.csv"
    
    try:
        with open(filename, 'w', newline='', encoding='utf-8') as f:
            f.write(data)
        
        print(f"Data saved to {filename}")
        return filename
    
    except IOError as e:
        print(f"Error saving file: {e}")
        return None

def main():
    """Main data pipeline"""
    print("Starting exoplanet data pipeline...")
    
    # Fetch data
    data = fetch_exoplanet_data()
    
    if data:
        # Save to CSV
        filename = save_csv_data(data)
        
        if filename:
            # Count rows (excluding header)
            rows = data.strip().split('\n')
            print(f"Successfully saved {len(rows) - 1} exoplanet records")
        else:
            print("Failed to save data")
    else:
        print("Failed to fetch data")

if __name__ == "__main__":
    main()
