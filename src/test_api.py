import os
import requests
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("NREL_API_KEY")

if not api_key:
    print("❌ Error: NREL_API_KEY not found in .env file")
else:
    # Testing with a sample coordinate (Atlanta, GA)
    url = "https://developer.nrel.gov/api/pvwatts/v8.json"
    params = {
        "api_key": api_key,
        "lat": 33.749,
        "lon": -84.388,
        "system_capacity": 4,
        "azimuth": 180,
        "tilt": 20,
        "array_type": 1,
        "module_type": 0,
        "losses": 14
    }
    
    try:
        response = requests.get(url, params=params)
        if response.status_code == 200:
            data = response.json()
            print("✅ Success! API key is working.")
            print(f"Annual Energy Production: {data['outputs']['ac_annual']:.2f} kWh")
        else:
            print(f"❌ API Request Failed: {response.status_code}")
            print(response.json().get('errors', 'No error message provided'))
    except Exception as e:
        print(f"❌ An error occurred: {e}")
