import requests
from datetime import datetime
import csv
from io import StringIO

def fetch_exoplanet_data():
    """
    """

    """Fetch exoplanet data from NASA Exoplanet Archive"""
    api = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync?query=select+hostname,pl_name,pl_rade,pl_masse,rowupdate+from+ps+where+pl_rade+<+=+1.8+and+pl_masse+>+0+and+disc_facility+=+'Transiting Exoplanet Survey Satellite (TESS)'&format=csv"
    
    try:
        print("Fetching exoplanet data...")
        response = requests.get(api, timeout=30)
        response.raise_for_status()
        
        return response.text
    
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data: {e}")
        return None

def save_csv_data(data, filename=None):
    """Save CSV data to file, deduplicated by hostname/pl_name keeping most recent rowupdate"""
    if not data:
        print("No data to save")
        return None
    
    if not filename:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"tess_exoplanets_{timestamp}.csv"
    
    try:
        # Parse CSV data
        csv_reader = csv.reader(StringIO(data))
        rows = list(csv_reader)
        
        if len(rows) < 2:  # No data rows to process
            with open(filename, 'w', newline='', encoding='utf-8') as f:
                f.write(data)
        else:
            header = rows[0]
            data_rows = rows[1:]
            
            # Create dictionary to store most recent entry for each hostname/pl_name combination
            unique_planets = {}
            original_count = len(data_rows)
            
            for row in data_rows:
                hostname = row[0]
                pl_name = row[1]
                rowupdate = row[4]
                
                key = (hostname, pl_name)
                
                # If this planet combination doesn't exist or this row has a more recent update
                if key not in unique_planets or rowupdate > unique_planets[key][4]:
                    unique_planets[key] = row
            
            # Convert back to list and sort
            deduplicated_rows = list(unique_planets.values())
            deduplicated_rows.sort(key=lambda x: (x[0], x[1], x[4]))
            
            # Write deduplicated and sorted data
            with open(filename, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(header)
                writer.writerows(deduplicated_rows)
            
            duplicates_removed = original_count - len(deduplicated_rows)
            print(f"Data saved to {filename}")
            print(f"Removed {duplicates_removed} duplicate entries, kept {len(deduplicated_rows)} unique planets")
        
        return filename
    
    except IOError as e:
        print(f"Error saving file: {e}")
        return None
    except Exception as e:
        print(f"Error processing CSV data: {e}")
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
            print("Data pipeline completed successfully")
        else:
            print("Failed to save data")
    else:
        print("Failed to fetch data")

if __name__ == "__main__":
    main()
