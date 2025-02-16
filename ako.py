import ctypes
import json
import sys

# Load the shared library
try:
    lib = ctypes.CDLL('./libhoujuu.so')
except OSError as e:
    print(f"Error loading library: {e}")
    sys.exit(1)

# Define the return types and argument types for the functions
lib.init_houjuu.restype = ctypes.c_void_p
lib.init_houjuu.argtypes = [ctypes.c_int]

lib.process_input.restype = ctypes.c_char_p
lib.process_input.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_bool]

lib.cleanup_houjuu.argtypes = [ctypes.c_void_p]

# Wrapper functions
def init_houjuu(id):
    state = lib.init_houjuu(id)
    if not state:
        raise RuntimeError("Failed to initialize houjuu state")
    return state

def process_input(state, input_str, agari_and_houjuu=False):
    if not state:
        raise ValueError("Invalid state pointer")
    input_bytes = input_str.encode('utf-8')
    result = lib.process_input(state, input_bytes, agari_and_houjuu)
    if result:
        return result.decode('utf-8')
        return json.loads(result.decode('utf-8'))
    return None

def cleanup_houjuu(state):
    if state:
        lib.cleanup_houjuu(state)

# Usage example
if __name__ == "__main__":
    state = None
    try:
        # Initialize the state
        state = init_houjuu(0)
        print("State initialized successfully")

        with open('mjai.json', 'r') as file:
            for line in file:
                input_str = line.strip()
                print(f"Processing input: {input_str[:50]}...")  # Print first 50 chars of input
                result = process_input(state, input_str, agari_and_houjuu=True)
                print(f"Result: {result}")

    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        # Clean up
        if state:
            cleanup_houjuu(state)
            print("Cleanup completed")