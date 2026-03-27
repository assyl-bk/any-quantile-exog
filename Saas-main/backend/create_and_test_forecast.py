#!/usr/bin/env python
"""Create API key and test the forecast API."""

import requests
import json
import numpy as np
from datetime import datetime

def create_user_and_keys():
    """Create a test user and API key."""
    print("=" * 60)
    print("CREATING TEST USER AND API KEY")
    print("=" * 60)
    
    from datetime import datetime as dt
    timestamp = dt.now().strftime('%Y%m%d_%H%M%S')
    
    # Signup
    signup_url = 'http://localhost:8000/api/auth/signup'
    signup_data = {
        'email': f'forecast_test_{timestamp}@example.com',
        'password': 'Test_Password_123',
        'name': 'Forecast Tester',
        'role': 'energy_trader'
    }
    
    print(f'\n1. Signing up user...')
    response = requests.post(signup_url, json=signup_data)
    if response.status_code not in [200, 201]:
        print(f'   Signup failed: {response.status_code}')
        print(f'   {response.text}')
        return None
    print(f'   ✓ User created')
    
    # Login
    login_url = 'http://localhost:8000/api/auth/login'
    login_data = {
        'email': 'forecast_test@example.com',
        'password': 'Test_Password_123'
    }
    print(f'\n2. Logging in...')
    response = requests.post(login_url, json=login_data)
    if response.status_code != 200:
        print(f'   Login failed: {response.status_code}')
        print(f'   {response.text}')
        return None
    
    login_result = response.json()
    token = login_result.get('access_token')
    print(f'   ✓ Logged in, token: {token[:20]}...')
    
    # Create API key
    keys_url = 'http://localhost:8000/api/keys/'
    key_data = {
        'name': 'Forecast Test Key',
        'description': 'Test forecast key'
    }
    headers = {
        'Authorization': f'Bearer {token}',
        'Content-Type': 'application/json'
    }
    
    print(f'\n3. Creating API key...')
    response = requests.post(keys_url, json=key_data, headers=headers)
    if response.status_code not in [200, 201]:
        print(f'   Key creation failed: {response.status_code}')
        print(f'   {response.text}')
        return None
    
    key_result = response.json()
    api_key = key_result.get('key')
    print(f'   ✓ API key created: {api_key[:20]}...')
    
    return api_key

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

def test_forecast(api_key):
    """Make a forecast request with the API key."""
    print("\n" + "=" * 60)
    print("TESTING FORECAST API")
    print("=" * 60)
    
    # Generate data
    historical_data = generate_historical_data(168)
    print(f'\nGenerated {len(historical_data)} hours of historical data')
    print(f'Sample values: {[f"{v:.2f}" for v in historical_data[:5]]}')
    print(f'Min: {min(historical_data):.2f}, Max: {max(historical_data):.2f}')
    
    # Make forecast request
    url = 'http://localhost:8000/api/forecast/forecast'
    payload = {
        'historical_data': historical_data,
        'quantiles': [0.1, 0.25, 0.5, 0.75, 0.9],
        'apply_cqr': False
    }
    
    headers = {
        'X-API-Key': api_key,
        'Content-Type': 'application/json'
    }
    
    print(f'\nMaking forecast request to: {url}')
    print(f'Historical data: {len(payload["historical_data"])} hours')
    print(f'Quantiles: {payload["quantiles"]}')
    print(f'Timestamp: {datetime.now().isoformat()}')
    
    print('\n⏳ Waiting for model inference...')
    try:
        response = requests.post(url, json=payload, headers=headers, timeout=120)
        
        print(f'\n✓ Response Status: {response.status_code}')
        
        if response.status_code == 200:
            result = response.json()
            print('\n' + "=" * 60)
            print('✅ FORECAST SUCCESSFUL!')
            print("=" * 60)
            
            print(f'\nModel Version: {result.get("model_version")}')
            print(f'CQR Applied: {result.get("cqr_applied")}')
            print(f'Forecast Horizon: {result.get("forecast_horizon")} hours')
            print(f'Quantiles: {result.get("quantiles")}')
            
            # Print forecast values
            if result.get('forecasts'):
                print('\n📊 First 5 hours of predictions:')
                for forecast in result['forecasts']:
                    quantile = forecast['quantile']
                    values = forecast['values'][:5]
                    formatted = [f'{v:.2f}' for v in values]
                    print(f'  Q{quantile:.2f}: {formatted}')
                
                print('\n📊 ALL forecast values (first 10 hours):')
                for forecast in result['forecasts']:
                    quantile = forecast['quantile']
                    values = forecast['values'][:10]
                    print(f'  Q{quantile:.2f}:')
                    for i, v in enumerate(values):
                        print(f'    Hour {i+1}: {v:.2f}')
            
            print(f'\n✨ Full Response:\n{json.dumps(result, indent=2)}')
        else:
            print(f'\n❌ Error {response.status_code}:')
            print(f'{response.text}')
    
    except Exception as e:
        print(f'\n❌ Exception: {e}')
        import traceback
        traceback.print_exc()

def main():
    print('\n🔧 ENERGY FORECAST MODEL INTEGRATION TEST\n')
    
    # Create user and API key
    api_key = create_user_and_keys()
    if not api_key:
        print('\n❌ Failed to create API key')
        return
    
    # Test forecast
    test_forecast(api_key)
    
    print('\n' + "=" * 60)
    print('✅ TEST COMPLETE')
    print("=" * 60)

if __name__ == '__main__':
    main()
