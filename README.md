# **imgcom**

**imgcom** is a powerful terminal CLI tool for macOS (and Linux/Windows) designed to combine multiple photos into one. It features a "Smart Mode" that uses computer vision to detect overlapping parts and stitch images seamlessly (handling any degree of rotation), as well as simple horizontal or vertical stacking modes.

## **Features**

* **Smart Stitching:** Automatically detects overlapping fields of view, corrects perspective, handles rotation, and blends images seamlessly.  
* **Simple Stacking:** Quickly stack images horizontally or vertically (great for screenshots, receipts, or document scans).  
* **Auto-Crop:** Automatically removes black borders generated during perspective warping.  
* **MacOS Integration:** Automatically opens the resulting image in Preview upon completion.

## **Installation**

### **Prerequisites**

You need Python 3 installed. You can install the required dependencies using pip:

pip install opencv-python numpy

### **Setup**

Clone the repository and make the script executable:

git clone \[https://github.com/makalin/imgcom.git\](https://github.com/makalin/imgcom.git)  
cd imgcom  
chmod \+x imgcom.py

## **Usage**

### **Smart Mode (Default)**

Best for panoramas or scanning large documents in parts. The tool will analyze features and stitch them together.

./imgcom.py part1.jpg part2.jpg part3.jpg

### **Horizontal Stack**

Combine images side-by-side without feature detection.

./imgcom.py left.png right.png \-m horizontal \-o combined.png

### **Vertical Stack**

Combine images top-to-bottom.

./imgcom.py top.jpg bottom.jpg \-m vertical

### **Options**

| Flag | Description |
| :---- | :---- |
| \-o, \--output | Specify the output filename. Defaults to imgcom\_result\_{timestamp}.jpg. |
| \-m, \--mode | Stitching mode: smart (default), horizontal, or vertical. |
| \--no-crop | Skip the auto-cropping of black borders in smart mode. |

## **Author**

**Mehmet T. AKALIN** *Digital Vision* [Website](https://dv.com.tr) | [LinkedIn](https://www.linkedin.com/in/makalin/) | [X (Twitter)](https://x.com/makalin)

## **License**

This project is licensed under the MIT License.
