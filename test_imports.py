"""
Test script to verify all required imports are available
Run this to check your environment setup
"""

import sys

def test_import(module_name, package_name=None):
    """Test if a module can be imported"""
    try:
        __import__(module_name)
        print(f"✓ {package_name or module_name}")
        return True
    except ImportError as e:
        print(f"✗ {package_name or module_name}: {e}")
        return False

print("=" * 60)
print("Testing Dependencies")
print("=" * 60)

results = {}

print("\nCore Libraries:")
results['streamlit'] = test_import('streamlit')
results['numpy'] = test_import('numpy')
results['pandas'] = test_import('pandas')

print("\nMachine Learning:")
results['sklearn'] = test_import('sklearn', 'scikit-learn')
results['tensorflow'] = test_import('tensorflow')

print("\nNLP:")
results['spacy'] = test_import('spacy')
results['textblob'] = test_import('textblob')

print("\nExplainability:")
results['lime'] = test_import('lime')
results['shap'] = test_import('shap')

print("\nVisualization:")
results['matplotlib'] = test_import('matplotlib')
results['seaborn'] = test_import('seaborn')
results['plotly'] = test_import('plotly')

print("\nStreamlit Components:")
results['canvas'] = test_import('streamlit_drawable_canvas', 'streamlit-drawable-canvas')

print("\nImage Processing:")
results['PIL'] = test_import('PIL', 'Pillow')
results['cv2'] = test_import('cv2', 'opencv-python')

print("\n" + "=" * 60)
passed = sum(results.values())
total = len(results)
print(f"Results: {passed}/{total} packages available")
print("=" * 60)

if passed == total:
    print("\n✓ All dependencies are installed! Ready to run the app.")
    print("  Run: streamlit run app.py")
else:
    print("\n⚠ Some dependencies are missing. Install them with:")
    print("  pip install -r requirements.txt")

    missing = [name for name, installed in results.items() if not installed]
    if 'spacy' in results and results['spacy']:
        print("\nDon't forget to download the spaCy model:")
        print("  python -m spacy download en_core_web_sm")

sys.exit(0 if passed == total else 1)
