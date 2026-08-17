import sys
import types

# Global test suite patch for Windows embedded Python lacking _bz2 C module
if '_bz2' not in sys.modules:
    try:
        import _bz2
    except ImportError:
        dummy_bz2 = types.ModuleType('_bz2')
        dummy_bz2.BZ2Compressor = object
        dummy_bz2.BZ2Decompressor = object
        sys.modules['_bz2'] = dummy_bz2
