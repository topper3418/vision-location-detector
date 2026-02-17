import sys
import importlib
import subprocess

REQUIRED_LIBRARIES = {
    'torch': '2.0.0',
    'ultralytics': '8.0.0',
    'tensorrt': '8.0.0',
    'opencv-python': '4.5.0',
    'numpy': '1.20.0',
}

# Optionally, add CUDA version check
CUDA_MIN_VERSION = '11.0'


def check_version(module_name, min_version):
    try:
        module = importlib.import_module(module_name)
        version = getattr(module, '__version__', None)
        if not version:
            print(f"[WARN] {module_name} has no __version__ attribute.")
            return False
        if tuple(map(int, version.split('.'))) < tuple(map(int, min_version.split('.'))):
            print(f"[FAIL] {module_name} version {version} < required {min_version}")
            return False
        print(f"[OK] {module_name} version {version} >= required {min_version}")
        return True
    except ImportError:
        print(f"[FAIL] {module_name} not installed.")
        return False


def check_cuda_version():
    try:
        output = subprocess.check_output(['nvcc', '--version'], encoding='utf-8')
        for line in output.split('\n'):
            if 'release' in line:
                version = line.split('release')[-1].strip().split(',')[0]
                if tuple(map(int, version.split('.'))) < tuple(map(int, CUDA_MIN_VERSION.split('.'))):
                    print(f"[FAIL] CUDA version {version} < required {CUDA_MIN_VERSION}")
                    return False
                print(f"[OK] CUDA version {version} >= required {CUDA_MIN_VERSION}")
                return True
        print("[WARN] Could not parse CUDA version.")
        return False
    except Exception as e:
        print(f"[FAIL] CUDA not found or nvcc not installed: {e}")
        return False


def main():
    print("--- Library Compatibility Check ---")
    all_ok = True
    for lib, min_ver in REQUIRED_LIBRARIES.items():
        ok = check_version(lib, min_ver)
        all_ok = all_ok and ok
    print("--- CUDA Compatibility Check ---")
    cuda_ok = check_cuda_version()
    all_ok = all_ok and cuda_ok
    if all_ok:
        print("\nAll required libraries and CUDA are compatible and up to date.")
    else:
        print("\nSome libraries or CUDA are missing or outdated. Please update.")

if __name__ == '__main__':
    main()
