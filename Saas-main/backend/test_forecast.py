#!/usr/bin/env python
"""Test script to call the forecast API with realistic energy data."""

import requests
import json
import numpy as np
from datetime import datetime

def generate_historical_data(hours=168):
    """Generate realistic 168-hour historical energy data."""
    base_load = 7000
    historical_data = []
    
    for i in range(hours):
        hour = i % 24
        day_of_week = i // 24
        
        # Daily pattern: Lower at night, peaks during business hours
        if 6 <= hour < 9:
            load = base_load + 2000  # Morning peak
        elif 9 <= hour < 17:
            load = base_load + 2500  # Business hours
        elif 17 <= hour < 21:
            load = base_load + 3000  # Evening peak
        elif 21 <= hour or hour < 6:
            load = base_load - 1000  # Night valley
        else:
            load = base_load
        
        # Weekend adjustment
        if day_of_week in [5, 6]:  # Saturday, Sunday
            load *= 0.85
        
        # Add realistic noise
        load += np.sin(i / 24) * 300 + np.random.randn() * 100
        
        historical_data.append(float(max(0, load)))
    
    return historical_data

def main():
    print("=" * 60)
    print("ENERGY FORECAST API TEST")
    print("=" * 60)
    
    # Generate data
    historical_data = generate_historical_data(168)
    print(f'\nGenerated {len(historical_data)} hours of historical data')
    print(f'Sample values: {[f"{v:.2f}" for v in historical_data[:5]]}')
    print(f'Min: {min(historical_data):.2f}, Max: {max(historical_data):.2f}')
    
    # Prepare request
    url = 'http://localhost:8000/api/forecast/forecast'
    payload = {
        'historical_data': historical_data,
        'quantiles': [0.1, 0.25, 0.5, 0.75, 0.9],
        'apply_cqr': False  # Disable CQR due to calibrator pickle issue
    }
    
    headers = {
        'X-API-Key': 'test-key-1',
        'Content-Type': 'application/json'
    }
    
    print('\n' + "=" * 60)
    print("MAKING FORECAST REQUEST")
    print("=" * 60)
    print(f'URL: {url}')
    print(f'Historical data length: {len(payload["historical_data"])}')
    print(f'Quantiles: {payload["quantiles"]}')
    print(f'Timestamp: {datetime.now().isoformat()}')
    
    try:
        print('\n⏳ Waiting for response...')
        response = requests.post(url, json=payload, headers=headers, timeout=120)
        
        print(f'\n✓ Response Status: {response.status_code}')
        print(f'Content-Type: {response.headers.get("content-type")}')
        
        if response.status_code == 200:
            result = response.json()
            print('\n' + "=" * 60)
            print("✅ FORECAST SUCCESSFUL!")
            print("=" * 60)
            
            print(f"\nModel Version: {result.get('model_version')}")
            print(f"CQR Applied: {result.get('cqr_applied')}")
            print(f"Forecast Horizon: {result.get('forecast_horizon')} hours")
            print(f"Quantiles: {result.get('quantiles')}")
            
            # Print first forecast values
            if result.get('forecasts'):
                print("\nFirst Forecast Values (first 5 hours):")
                for forecast in result['forecasts']:
                    quantile = forecast['quantile']
                    values = forecast['values'][:5]
                    print(f"  Q{quantile:.2f}: {[f'{v:.2f}' for v in values]}")
            
            # Print full response
            print(f'\nFull Response:\n{json.dumps(result, indent=2)}')
        else:
            print(f'\n❌ Error Response:')
            print(f'Status: {response.status_code}')
            print(f'Body: {response.text}')
    
    except requests.exceptions.ConnectionError as e:
        print(f'\n❌ CONNECTION ERROR: Could not connect to {url}')
        print(f'Make sure the backend server is running on port 8000')
        print(f'Error: {e}')
    except Exception as e:
        print(f'\n❌ ERROR: {e}')
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()
