# **imgcom** - Professional Image Combiner & Stitcher

**imgcom** is a powerful, professional-grade terminal CLI tool for macOS, Linux, and Windows designed to combine multiple photos into one. It features advanced computer vision stitching, multiple output formats, batch processing, image preprocessing, watermarking, and much more.

## **Features**

### **Core Stitching Modes**
* **Smart Stitching:** Automatically detects overlapping fields of view, corrects perspective, handles rotation, and blends images seamlessly using feature detection.
* **Simple Stacking:** Quickly stack images horizontally or vertically (great for screenshots, receipts, or document scans).
* **Grid/Mosaic Mode:** Arrange multiple images in a customizable grid layout with spacing and background options.
* **Auto-Crop:** Automatically removes black borders generated during perspective warping.

### **Professional Tools**
* **Multiple Output Formats:** Support for JPG, PNG, WebP, and TIFF with adjustable quality settings.
* **Image Preprocessing:** Resize, rotate, adjust brightness/contrast, and apply filters before combining.
* **Batch Processing:** Process multiple image sets automatically with JSON configuration.
* **Watermarking:** Add text or image watermarks with customizable position and opacity.
* **EXIF Preservation:** Preserve image metadata when using Pillow (optional).
* **Image Statistics:** Display detailed information about images (dimensions, color stats, etc.).
* **Advanced Cropping:** Custom crop coordinates and automatic border removal.
* **Verbose Mode:** Detailed logging and progress indicators for debugging.
* **macOS Integration:** Automatically opens the resulting image in Preview upon completion.

## **Installation**

### **Prerequisites**

You need Python 3.6+ installed. Install the required dependencies:

```bash
pip install opencv-python numpy
```

For advanced features (EXIF preservation, enhanced watermarking):
```bash
pip install Pillow
```

### **Setup**

Clone the repository and make the script executable:

```bash
git clone https://github.com/makalin/imgcom.git
cd imgcom
chmod +x imgcom.py
```

Or install globally:
```bash
sudo cp imgcom.py /usr/local/bin/imgcom
sudo chmod +x /usr/local/bin/imgcom
```

## **Usage**

### **Smart Mode (Default)**

Best for panoramas or scanning large documents in parts. The tool analyzes features and stitches them together.

```bash
./imgcom.py part1.jpg part2.jpg part3.jpg
```

### **Horizontal Stack**

Combine images side-by-side without feature detection.

```bash
./imgcom.py left.png right.png -m horizontal -o combined.png
```

### **Vertical Stack**

Combine images top-to-bottom.

```bash
./imgcom.py top.jpg bottom.jpg -m vertical
```

### **Grid Layout**

Arrange images in a grid (e.g., 3 columns with 10px spacing):

```bash
./imgcom.py *.jpg -m grid --grid-cols 3 --grid-spacing 10
```

### **With Preprocessing**

Resize, adjust brightness/contrast, and apply filters before combining:

```bash
./imgcom.py img1.jpg img2.jpg --resize 1920 1080 --brightness 10 --contrast 5
```

### **Add Watermark**

Add a text watermark:

```bash
./imgcom.py img1.jpg img2.jpg --watermark "© 2024" --watermark-pos top-right
```

Add an image watermark:

```bash
./imgcom.py img1.jpg img2.jpg --watermark-img logo.png --watermark-opacity 0.7
```

### **Batch Processing**

Process multiple image sets using a JSON configuration file:

```bash
./imgcom.py --batch batch_config.json --batch-dir output
```

Example `batch_config.json`:
```json
{
  "mode": "smart",
  "quality": 95,
  "grid_cols": 3,
  "grid_spacing": 5,
  "sets": [
    ["img1.jpg", "img2.jpg", "img3.jpg"],
    ["img4.jpg", "img5.jpg"],
    ["img6.jpg", "img7.jpg", "img8.jpg", "img9.jpg"]
  ]
}
```

### **Image Information**

Display detailed information about images:

```bash
./imgcom.py img1.jpg img2.jpg --info
```

### **Custom Output Format & Quality**

Save as PNG with custom quality:

```bash
./imgcom.py img1.jpg img2.jpg --format png --quality 100 -o result.png
```

## **Options**

### **Input/Output**
| Flag | Description |
| :---- | :---- |
| `-o, --output` | Specify the output filename. Defaults to `imgcom_result_{timestamp}.jpg`. |
| `--format` | Output format: `jpg`, `png`, `webp`, or `tiff`. Auto-detected from filename if not specified. |
| `--quality` | Output quality 1-100 (default: 95). Higher values = better quality, larger files. |

### **Modes**
| Flag | Description |
| :---- | :---- |
| `-m, --mode` | Combination mode: `smart` (default), `horizontal`, `vertical`, or `grid`. |

### **Grid Options**
| Flag | Description |
| :---- | :---- |
| `--grid-cols` | Number of columns for grid mode (default: 2). |
| `--grid-spacing` | Spacing between images in pixels (default: 0). |
| `--grid-bg R G B` | Background color RGB values (default: 0 0 0). |

### **Preprocessing**
| Flag | Description |
| :---- | :---- |
| `--resize WIDTH HEIGHT` | Resize all images before processing. |
| `--rotate DEGREES` | Rotate all images by specified degrees. |
| `--brightness VALUE` | Adjust brightness (-100 to 100). |
| `--contrast VALUE` | Adjust contrast (-100 to 100). |
| `--filter TYPE` | Apply filter: `blur`, `sharpen`, or `edge`. |

### **Cropping**
| Flag | Description |
| :---- | :---- |
| `--no-crop` | Skip auto-cropping black borders in smart mode. |
| `--crop X Y W H` | Custom crop after processing (x, y, width, height). |

### **Watermarking**
| Flag | Description |
| :---- | :---- |
| `--watermark TEXT` | Add text watermark. |
| `--watermark-img PATH` | Add image watermark from file. |
| `--watermark-pos POS` | Watermark position: `top-left`, `top-right`, `bottom-left`, `bottom-right` (default: `bottom-right`). |
| `--watermark-opacity VALUE` | Watermark opacity 0.0-1.0 (default: 0.5). |
| `--watermark-scale VALUE` | Image watermark scale factor (default: 0.1). |

### **Batch Processing**
| Flag | Description |
| :---- | :---- |
| `--batch FILE` | JSON file with batch processing configuration. |
| `--batch-dir DIR` | Output directory for batch processing (default: `output`). |

### **Utilities**
| Flag | Description |
| :---- | :---- |
| `--info` | Display detailed image information. |
| `--preserve-exif` | Preserve EXIF data (requires Pillow). |
| `-v, --verbose` | Verbose output with detailed logging. |
| `--no-preview` | Do not open result in Preview (macOS). |

## **Examples**

### **Create a Panorama**
```bash
./imgcom.py photo1.jpg photo2.jpg photo3.jpg -o panorama.jpg
```

### **Combine Screenshots Horizontally**
```bash
./imgcom.py screen1.png screen2.png screen3.png -m horizontal -o combined.png
```

### **Create a Photo Grid**
```bash
./imgcom.py photo*.jpg -m grid --grid-cols 4 --grid-spacing 5 --grid-bg 255 255 255 -o grid.jpg
```

### **Process with Enhancements**
```bash
./imgcom.py img1.jpg img2.jpg \
  --resize 1920 1080 \
  --brightness 15 \
  --contrast 10 \
  --watermark "My Photography" \
  --watermark-pos bottom-right \
  --format png \
  --quality 100 \
  -o enhanced.png
```

### **Batch Process Multiple Sets**
```bash
# Create batch_config.json with your image sets
./imgcom.py --batch batch_config.json --batch-dir results -v
```

## **Advanced Usage**

### **Glob Patterns**
Use wildcards to select multiple images:

```bash
./imgcom.py images/*.jpg -m grid --grid-cols 3
```

### **Verbose Mode**
Get detailed information about the processing:

```bash
./imgcom.py img1.jpg img2.jpg -v
```

### **Custom Crop**
After stitching, crop to specific coordinates:

```bash
./imgcom.py img1.jpg img2.jpg --crop 100 100 800 600 -o cropped.jpg
```

## **Tips**

* **Smart Mode:** Ensure images have at least 30% visual overlap for best results.
* **Quality Settings:** Use quality 95-100 for photos, 80-90 for web images, 60-80 for thumbnails.
* **Grid Mode:** Use `--grid-spacing` to add visual separation between images.
* **Batch Processing:** Perfect for processing multiple panoramas or image sets automatically.
* **Preprocessing:** Apply filters and adjustments before combining for consistent results.

## **Requirements**

* Python 3.6+
* OpenCV (`opencv-python`)
* NumPy
* Pillow (optional, for EXIF preservation and advanced features)

## **Author**

**Mehmet T. AKALIN** *Digital Vision*  
[Website](https://dv.com.tr) | [LinkedIn](https://www.linkedin.com/in/makalin/) | [X (Twitter)](https://x.com/makalin)

## **License**

This project is licensed under the MIT License.
