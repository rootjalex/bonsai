import sys
from PIL import Image, UnidentifiedImageError
import numpy as np
import os

def convert_ppm_to_png(input_path, output_path):
    try:
        img = Image.open(input_path)
    except FileNotFoundError:
        print(f"Error: File not found - {input_path}")
        sys.exit(1)
    except UnidentifiedImageError:
        print(f"Error: Cannot identify image file - {input_path}")
        sys.exit(1)
    except Exception as e:
        print(f"Error: Failed to open input file - {e}")
        sys.exit(1)

    try:
        arr = np.array(img)
        img_out = Image.fromarray(arr)
        img_out.save(output_path)
        print(f"Saved: {output_path}")
    except OSError as e:
        print(f"Error: Failed to save output file - {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error: Unexpected error during saving - {e}")
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python ppm_to_png.py <input.ppm> <output.png>")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]
    convert_ppm_to_png(input_file, output_file)
