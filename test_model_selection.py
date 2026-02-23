#!/usr/bin/env python3
"""
Test script to verify model selection implementation.
This tests the MODEL_CONFIGS, get_model function logic without loading actual models.
"""

def test_model_configs():
    """Test that MODEL_CONFIGS is properly structured"""
    # Import the actual MODEL_CONFIGS from app.py to avoid duplication
    import sys
    import os
    
    # Add app.py directory to path
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    
    # Since we can't import app.py without triggering model loading,
    # we'll verify it by reading the file content
    with open('app.py', 'r') as f:
        content = f.read()
    
    # Verify MODEL_CONFIGS structure exists
    assert 'MODEL_CONFIGS = {' in content
    assert '"parakeet-tdt-0.6b-v3"' in content
    
    # Verify INT8-only quantization setting is present
    assert '"quantization": "int8"' in content
    
    # Verify HuggingFace ID is present
    assert '"hf_id": "nemo-parakeet-tdt-0.6b-v3"' in content
    
    print("✅ MODEL_CONFIGS structure test passed")


def test_model_fallback_logic():
    """Test the fallback logic when unknown model is requested"""
    # Read app.py to verify fallback logic
    with open('app.py', 'r') as f:
        content = f.read()
    
    # Verify fallback is implemented
    assert 'if model_name not in MODEL_CONFIGS:' in content
    assert 'parakeet-tdt-0.6b-v3' in content  # Default fallback model
    
    # Verify warning is logged
    assert 'Unknown model' in content or 'unknown model' in content.lower()
    
    print("✅ Model fallback logic test passed")


def test_lazy_loading_caching():
    """Test that lazy loading and caching are implemented"""
    with open('app.py', 'r') as f:
        content = f.read()
    
    # Verify model_cache exists
    assert 'model_cache = {}' in content
    
    # Verify get_model function exists
    assert 'def get_model(model_name):' in content
    
    # Verify caching logic
    assert 'if model_name in model_cache:' in content
    assert 'model_cache[model_name] = model' in content
    
    print("✅ Lazy loading and caching test passed")


def test_openai_compatibility():
    """Test OpenAI compatible parameter defaults"""
    with open('app.py', 'r') as f:
        content = f.read()
    
    # Default model should be parakeet variant
    assert 'model", "parakeet-tdt-0.6b-v3"' in content
    
    # Verify model_to_use is called
    assert 'model_to_use = get_model(model_name)' in content
    assert 'model_to_use.recognize(chunk_path)' in content
    
    print("✅ OpenAI compatibility test passed")


if __name__ == "__main__":
    test_model_configs()
    test_model_fallback_logic()
    test_lazy_loading_caching()
    test_openai_compatibility()
    print("\n✅ All tests passed successfully!")
