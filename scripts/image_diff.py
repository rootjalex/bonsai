import sys
from PIL import Image, UnidentifiedImageError
import numpy as np

def load_image(path):
    try:
        return Image.open(path).convert("RGB")
    except FileNotFoundError:
        print(f"Error: File not found - {path}")
        sys.exit(1)
    except UnidentifiedImageError:
        print(f"Error: Cannot identify image file - {path}")
        sys.exit(1)
    except Exception as e:
        print(f"Error: Failed to open {path} - {e}")
        sys.exit(1)

def main():
    if len(sys.argv) not in [4, 5]:
        print("Usage: python image_diff.py <image0.ppm> <image1.ppm> <output.ppm> [scale]")
        sys.exit(1)

    path0, path1, output_path = sys.argv[1], sys.argv[2], sys.argv[3]
    try:
        scale = float(sys.argv[4]) if len(sys.argv) == 5 else 1.0
    except ValueError:
        print("Error: Scale must be a numeric value")
        sys.exit(1)

    img0 = load_image(path0)
    img1 = load_image(path1)

    if img0.size != img1.size:
        print("Error: Images must be the same size")
        sys.exit(1)

    arr0 = np.array(img0, dtype=np.int16)
    arr1 = np.array(img1, dtype=np.int16)

    diff = np.abs(arr0 - arr1) * scale
    diff = np.clip(diff, 0, 255).astype(np.uint8)

    diff_img = Image.fromarray(diff, mode="RGB")

    try:
        diff_img.save(output_path)
        print(f"Saved: {output_path}")
    except OSError as e:
        print(f"Error: Failed to save output file - {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error: Unexpected error during saving - {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
