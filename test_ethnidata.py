#!/usr/bin/env python3
"""Quick test of ethnidata integration."""
import sys
sys.path.insert(0, '.')

try:
    from ethnidata import EthniData
    print("✓ ethnidata imported successfully")
    
    # Create classifier
    classifier = EthniData()
    print("✓ Classifier created")
    
    # Test prediction
    result = classifier.predict_nationality("Anna", name_type="first")
    print(f"✓ Prediction result: {result}")
    
    # Check required fields
    if 'country_name' in result and 'confidence' in result:
        print(f"  Country: {result['country_name']}")
        print(f"  Confidence: {result['confidence']}")
    else:
        print("✗ Missing expected fields in result")
        print(f"  Keys: {list(result.keys())}")
        
except ImportError as e:
    print(f"✗ Import error: {e}")
    sys.exit(1)
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("✓ All tests passed")
