#!/usr/bin/env python3
"""
imgcom - Professional Smart Image Combiner & Stitcher CLI
Author: Mehmet T. AKALIN
"""

import cv2
import numpy as np
import argparse
import sys
import os
import json
import glob
from datetime import datetime
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any

try:
    from PIL import Image, ImageDraw, ImageFont, ExifTags
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    print("[!] Warning: PIL/Pillow not installed. Some features (EXIF, advanced watermarking) will be limited.")
    print("    Install with: pip install Pillow")

class ImgCom:
    def __init__(self, verbose=False):
        self.verbose = verbose
        # Initialize OpenCV's stitcher
        try:
            self.stitcher = cv2.Stitcher_create() if hasattr(cv2, 'Stitcher_create') else cv2.createStitcher(False)
        except:
            self.stitcher = None
            if verbose:
                print("[!] Warning: Stitcher initialization failed")

    def log(self, message, level="INFO"):
        """Logging helper"""
        if self.verbose or level in ["ERROR", "WARNING"]:
            prefix = {"INFO": "[*]", "SUCCESS": "[+]", "ERROR": "[!]", "WARNING": "[~]"}
            print(f"{prefix.get(level, '[*]')} {message}")

    def load_images(self, image_paths: List[str], preprocess: Optional[Dict] = None) -> List[np.ndarray]:
        """Loads and optionally preprocesses images from paths."""
        images = []
        total = len(image_paths)
        self.log(f"Loading {total} images...")
        
        for idx, path in enumerate(image_paths, 1):
            if not os.path.exists(path):
                self.log(f"File not found: {path}", "ERROR")
                continue
            
            img = cv2.imread(path)
            if img is None:
                self.log(f"Could not decode image: {path}", "ERROR")
                continue
            
            # Apply preprocessing if specified
            if preprocess:
                img = self.preprocess_image(img, preprocess, idx)
            
            images.append(img)
            if self.verbose:
                h, w = img.shape[:2]
                self.log(f"  [{idx}/{total}] Loaded {os.path.basename(path)} ({w}x{h})")
        
        return images

    def preprocess_image(self, img: np.ndarray, options: Dict, index: int = 0) -> np.ndarray:
        """Apply preprocessing options to an image."""
        result = img.copy()
        
        # Resize
        if 'resize' in options:
            width, height = options['resize']
            result = cv2.resize(result, (width, height), interpolation=cv2.INTER_LANCZOS4)
            self.log(f"  Resized image {index} to {width}x{height}")
        
        # Rotate
        if 'rotate' in options:
            angle = options['rotate']
            h, w = result.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            result = cv2.warpAffine(result, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
            self.log(f"  Rotated image {index} by {angle}°")
        
        # Brightness adjustment
        if 'brightness' in options:
            beta = int(options['brightness'] * 255 / 100)
            result = cv2.convertScaleAbs(result, alpha=1, beta=beta)
            self.log(f"  Adjusted brightness of image {index} by {options['brightness']}%")
        
        # Contrast adjustment
        if 'contrast' in options:
            alpha = 1 + (options['contrast'] / 100)
            result = cv2.convertScaleAbs(result, alpha=alpha, beta=0)
            self.log(f"  Adjusted contrast of image {index} by {options['contrast']}%")
        
        # Apply filters
        if 'filter' in options:
            filter_type = options['filter']
            if filter_type == 'blur':
                result = cv2.GaussianBlur(result, (5, 5), 0)
            elif filter_type == 'sharpen':
                kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
                result = cv2.filter2D(result, -1, kernel)
            elif filter_type == 'edge':
                result = cv2.Canny(cv2.cvtColor(result, cv2.COLOR_BGR2GRAY), 100, 200)
                result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
        
        return result

    def smart_stitch(self, images: List[np.ndarray]) -> Optional[np.ndarray]:
        """Smart stitching using feature detection."""
        if not self.stitcher:
            self.log("Stitcher not available", "ERROR")
            return None
        
        self.log("Analyzing image features and stitching (Smart Mode)...")
        self.log("This may take a moment depending on image size and overlap.")
        
        status, stitched = self.stitcher.stitch(images)
        
        if status == cv2.Stitcher_OK:
            self.log("Smart stitch successful!", "SUCCESS")
            return stitched
        else:
            error_map = {
                cv2.Stitcher_ERR_NEED_MORE_IMGS: "Need more images",
                cv2.Stitcher_ERR_HOMOGRAPHY_EST_FAIL: "Homography estimation failed (not enough overlap?)",
                cv2.Stitcher_ERR_CAMERA_PARAMS_ADJUST_FAIL: "Camera parameters adjustment failed"
            }
            err_msg = error_map.get(status, f"Unknown error code {status}")
            self.log(f"Smart stitching failed: {err_msg}", "ERROR")
            self.log("Tip: Ensure images have at least 30% visual overlap.")
            return None

    def simple_stitch(self, images: List[np.ndarray], direction: str = 'horizontal') -> Optional[np.ndarray]:
        """Simple concatenation of images."""
        self.log(f"Performing simple {direction} concatenation...")
        
        if not images:
            return None
        
        # Resize logic
        base_h, base_w = images[0].shape[:2]
        resized_images = [images[0]]
        
        for img in images[1:]:
            h, w = img.shape[:2]
            if direction == 'horizontal':
                new_w = int(w * (base_h / h))
                resized = cv2.resize(img, (new_w, base_h), interpolation=cv2.INTER_LANCZOS4)
            else:  # vertical
                new_h = int(h * (base_w / w))
                resized = cv2.resize(img, (base_w, new_h), interpolation=cv2.INTER_LANCZOS4)
            
            resized_images.append(resized)
        
        if direction == 'horizontal':
            return np.hstack(resized_images)
        else:
            return np.vstack(resized_images)

    def grid_stitch(self, images: List[np.ndarray], cols: int = 2, spacing: int = 0, 
                   bg_color: Tuple[int, int, int] = (0, 0, 0)) -> Optional[np.ndarray]:
        """Arrange images in a grid layout."""
        self.log(f"Creating {cols}-column grid layout...")
        
        if not images:
            return None
        
        rows = (len(images) + cols - 1) // cols
        
        # Resize all images to same size
        target_h = min(img.shape[0] for img in images)
        target_w = min(img.shape[1] for img in images)
        
        resized = []
        for img in images:
            resized_img = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)
            resized.append(resized_img)
        
        # Create grid
        cell_h = target_h + spacing
        cell_w = target_w + spacing
        grid_h = rows * cell_h - spacing
        grid_w = cols * cell_w - spacing
        
        grid = np.full((grid_h, grid_w, 3), bg_color, dtype=np.uint8)
        
        for idx, img in enumerate(resized):
            row = idx // cols
            col = idx % cols
            y = row * cell_h
            x = col * cell_w
            grid[y:y+target_h, x:x+target_w] = img
        
        return grid

    def crop_black_borders(self, img: np.ndarray, threshold: int = 1) -> np.ndarray:
        """Remove black borders from image."""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            x, y, w, h = cv2.boundingRect(contours[0])
            return img[y:y+h, x:x+w]
        return img

    def crop_custom(self, img: np.ndarray, x: int, y: int, w: int, h: int) -> np.ndarray:
        """Custom crop with coordinates."""
        h_img, w_img = img.shape[:2]
        x = max(0, min(x, w_img))
        y = max(0, min(y, h_img))
        w = min(w, w_img - x)
        h = min(h, h_img - y)
        return img[y:y+h, x:x+w]

    def add_watermark(self, img: np.ndarray, text: str, position: str = 'bottom-right',
                     opacity: float = 0.5, font_scale: float = 1.0) -> np.ndarray:
        """Add text watermark to image."""
        result = img.copy()
        h, w = result.shape[:2]
        
        # Get text size
        font = cv2.FONT_HERSHEY_SIMPLEX
        thickness = max(1, int(font_scale))
        (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
        
        # Calculate position
        margin = 10
        if position == 'top-left':
            x, y = margin, text_h + margin
        elif position == 'top-right':
            x, y = w - text_w - margin, text_h + margin
        elif position == 'bottom-left':
            x, y = margin, h - baseline - margin
        else:  # bottom-right (default)
            x, y = w - text_w - margin, h - baseline - margin
        
        # Create overlay
        overlay = result.copy()
        cv2.putText(overlay, text, (x, y), font, font_scale, (255, 255, 255), thickness + 2)
        cv2.putText(overlay, text, (x, y), font, font_scale, (0, 0, 0), thickness)
        
        # Blend
        result = cv2.addWeighted(overlay, opacity, result, 1 - opacity, 0)
        return result

    def add_image_watermark(self, img: np.ndarray, watermark_path: str, 
                           position: str = 'bottom-right', opacity: float = 0.5,
                           scale: float = 0.1) -> np.ndarray:
        """Add image watermark."""
        if not os.path.exists(watermark_path):
            self.log(f"Watermark image not found: {watermark_path}", "ERROR")
            return img
        
        watermark = cv2.imread(watermark_path, cv2.IMREAD_UNCHANGED)
        if watermark is None:
            return img
        
        h_img, w_img = img.shape[:2]
        h_wm, w_wm = watermark.shape[:2]
        
        # Resize watermark
        new_w = int(w_img * scale)
        new_h = int(h_wm * (new_w / w_wm))
        watermark = cv2.resize(watermark, (new_w, new_h))
        
        # Handle alpha channel
        if watermark.shape[2] == 4:
            alpha = watermark[:, :, 3] / 255.0
            watermark = watermark[:, :, :3]
        else:
            alpha = np.ones((new_h, new_w))
        
        # Calculate position
        margin = 10
        if position == 'top-left':
            y1, y2 = margin, margin + new_h
            x1, x2 = margin, margin + new_w
        elif position == 'top-right':
            y1, y2 = margin, margin + new_h
            x1, x2 = w_img - new_w - margin, w_img - margin
        elif position == 'bottom-left':
            y1, y2 = h_img - new_h - margin, h_img - margin
            x1, x2 = margin, margin + new_w
        else:  # bottom-right
            y1, y2 = h_img - new_h - margin, h_img - margin
            x1, x2 = w_img - new_w - margin, w_img - margin
        
        # Blend watermark
        result = img.copy()
        for c in range(3):
            result[y1:y2, x1:x2, c] = (alpha * watermark[:, :, c] * opacity + 
                                      result[y1:y2, x1:x2, c] * (1 - alpha * opacity))
        
        return result

    def get_image_info(self, img: np.ndarray) -> Dict[str, Any]:
        """Get statistics and info about image."""
        h, w = img.shape[:2]
        channels = img.shape[2] if len(img.shape) > 2 else 1
        
        # Calculate statistics
        mean = np.mean(img)
        std = np.std(img)
        
        # Color statistics
        if channels == 3:
            b, g, r = cv2.split(img)
            mean_b, mean_g, mean_r = np.mean(b), np.mean(g), np.mean(r)
        else:
            mean_b = mean_g = mean_r = mean
        
        return {
            'width': w,
            'height': h,
            'channels': channels,
            'mean': mean,
            'std': std,
            'mean_b': mean_b,
            'mean_g': mean_g,
            'mean_r': mean_r
        }

    def save_image(self, img: np.ndarray, output_path: str, quality: int = 95, 
                  preserve_exif: bool = False) -> bool:
        """Save image with format detection and quality options."""
        ext = os.path.splitext(output_path)[1].lower()
        
        # Determine save parameters based on format
        params = []
        if ext in ['.jpg', '.jpeg']:
            params = [cv2.IMWRITE_JPEG_QUALITY, quality]
        elif ext == '.png':
            params = [cv2.IMWRITE_PNG_COMPRESSION, 9 - (quality // 10)]
        elif ext == '.webp':
            params = [cv2.IMWRITE_WEBP_QUALITY, quality]
        elif ext == '.tiff':
            params = [cv2.IMWRITE_TIFF_COMPRESSION, 1]
        
        success = cv2.imwrite(output_path, img, params)
        
        if success and preserve_exif and PIL_AVAILABLE:
            # Try to preserve EXIF from first input image
            try:
                pil_img = Image.open(output_path)
                # EXIF preservation would go here if we had source images
                pil_img.save(output_path, quality=quality)
            except:
                pass
        
        return success

    def batch_process(self, image_sets: List[List[str]], output_dir: str = "output",
                     mode: str = 'smart', **kwargs) -> List[str]:
        """Process multiple sets of images."""
        os.makedirs(output_dir, exist_ok=True)
        results = []
        
        self.log(f"Batch processing {len(image_sets)} image sets...")
        
        for idx, image_set in enumerate(image_sets, 1):
            self.log(f"\nProcessing set {idx}/{len(image_sets)}...")
            
            images = self.load_images(image_set, kwargs.get('preprocess'))
            if len(images) < 2:
                self.log(f"Skipping set {idx}: need at least 2 images", "WARNING")
                continue
            
            # Process based on mode
            result = None
            if mode == 'smart':
                result = self.smart_stitch(images)
                if result is not None and not kwargs.get('no_crop'):
                    result = self.crop_black_borders(result)
            elif mode == 'grid':
                cols = kwargs.get('grid_cols', 2)
                spacing = kwargs.get('grid_spacing', 0)
                result = self.grid_stitch(images, cols=cols, spacing=spacing)
            else:
                result = self.simple_stitch(images, direction=mode)
            
            if result is not None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_path = os.path.join(output_dir, f"batch_{idx}_{timestamp}.jpg")
                
                if self.save_image(result, output_path, quality=kwargs.get('quality', 95)):
                    results.append(output_path)
                    self.log(f"Saved: {output_path}", "SUCCESS")
        
        return results


def main():
    parser = argparse.ArgumentParser(
        description="imgcom: Professional Image Combiner & Stitcher CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Smart stitch panorama
  %(prog)s img1.jpg img2.jpg img3.jpg
  
  # Horizontal stack with custom output
  %(prog)s left.png right.png -m horizontal -o combined.png
  
  # Grid layout
  %(prog)s *.jpg -m grid --grid-cols 3 --grid-spacing 10
  
  # With preprocessing
  %(prog)s img1.jpg img2.jpg --resize 1920 1080 --brightness 10 --contrast 5
  
  # Batch process multiple sets
  %(prog)s --batch batch_config.json
  
  # Add watermark
  %(prog)s img1.jpg img2.jpg --watermark "© 2024" --watermark-pos top-right
        """
    )
    
    # Input/output
    parser.add_argument('inputs', nargs='*', help='Image files to combine (or use --batch)')
    parser.add_argument('-o', '--output', help='Output filename', default=None)
    parser.add_argument('--format', choices=['jpg', 'png', 'webp', 'tiff'], 
                       help='Output format (default: auto-detect from output filename)')
    parser.add_argument('--quality', type=int, default=95, choices=range(1, 101),
                       help='Output quality 1-100 (default: 95)')
    
    # Modes
    parser.add_argument('-m', '--mode', choices=['smart', 'horizontal', 'vertical', 'grid'],
                       default='smart', help='Combination mode (default: smart)')
    
    # Grid options
    parser.add_argument('--grid-cols', type=int, default=2, help='Number of columns for grid mode')
    parser.add_argument('--grid-spacing', type=int, default=0, help='Spacing between images in grid (pixels)')
    parser.add_argument('--grid-bg', nargs=3, type=int, metavar=('R', 'G', 'B'),
                       default=[0, 0, 0], help='Grid background color RGB (default: 0 0 0)')
    
    # Preprocessing
    parser.add_argument('--resize', nargs=2, type=int, metavar=('WIDTH', 'HEIGHT'),
                       help='Resize all images before processing')
    parser.add_argument('--rotate', type=float, help='Rotate all images (degrees)')
    parser.add_argument('--brightness', type=int, help='Adjust brightness (-100 to 100)')
    parser.add_argument('--contrast', type=int, help='Adjust contrast (-100 to 100)')
    parser.add_argument('--filter', choices=['blur', 'sharpen', 'edge'],
                       help='Apply filter to images')
    
    # Cropping
    parser.add_argument('--no-crop', action='store_true',
                       help='Skip auto-cropping black borders in smart mode')
    parser.add_argument('--crop', nargs=4, type=int, metavar=('X', 'Y', 'W', 'H'),
                       help='Custom crop after processing (x, y, width, height)')
    
    # Watermarking
    parser.add_argument('--watermark', help='Text watermark to add')
    parser.add_argument('--watermark-img', help='Image watermark file path')
    parser.add_argument('--watermark-pos', choices=['top-left', 'top-right', 'bottom-left', 'bottom-right'],
                       default='bottom-right', help='Watermark position')
    parser.add_argument('--watermark-opacity', type=float, default=0.5,
                       help='Watermark opacity 0.0-1.0 (default: 0.5)')
    parser.add_argument('--watermark-scale', type=float, default=0.1,
                       help='Image watermark scale factor (default: 0.1)')
    
    # Batch processing
    parser.add_argument('--batch', help='JSON file with batch processing configuration')
    parser.add_argument('--batch-dir', default='output', help='Output directory for batch processing')
    
    # Info and utilities
    parser.add_argument('--info', action='store_true', help='Display image information')
    parser.add_argument('--preserve-exif', action='store_true', help='Preserve EXIF data (requires Pillow)')
    parser.add_argument('-v', '--verbose', action='store_true', help='Verbose output')
    parser.add_argument('--no-preview', action='store_true', help='Do not open result in Preview (macOS)')
    
    args = parser.parse_args()
    
    # Handle batch processing
    if args.batch:
        if not os.path.exists(args.batch):
            print(f"[!] Error: Batch config file not found: {args.batch}")
            sys.exit(1)
        
        with open(args.batch, 'r') as f:
            batch_config = json.load(f)
        
        tool = ImgCom(verbose=args.verbose)
        
        image_sets = batch_config.get('sets', [])
        mode = batch_config.get('mode', 'smart')
        
        results = tool.batch_process(
            image_sets,
            output_dir=args.batch_dir,
            mode=mode,
            grid_cols=batch_config.get('grid_cols', 2),
            grid_spacing=batch_config.get('grid_spacing', 0),
            quality=batch_config.get('quality', 95),
            no_crop=batch_config.get('no_crop', False)
        )
        
        print(f"\n[SUCCESS] Batch processing complete! {len(results)} images created.")
        sys.exit(0)
    
    # Regular processing
    if not args.inputs:
        parser.print_help()
        sys.exit(1)
    
    # Expand glob patterns
    image_paths = []
    for pattern in args.inputs:
        if '*' in pattern or '?' in pattern:
            image_paths.extend(glob.glob(pattern))
        else:
            image_paths.append(pattern)
    
    if not image_paths:
        print("[!] Error: No valid image files found.")
        sys.exit(1)
    
    tool = ImgCom(verbose=args.verbose)
    
    # Build preprocessing options
    preprocess = {}
    if args.resize:
        preprocess['resize'] = args.resize
    if args.rotate:
        preprocess['rotate'] = args.rotate
    if args.brightness:
        preprocess['brightness'] = args.brightness
    if args.contrast:
        preprocess['contrast'] = args.contrast
    if args.filter:
        preprocess['filter'] = args.filter
    
    # Load images
    images = tool.load_images(image_paths, preprocess if preprocess else None)
    
    if len(images) < 2:
        print("[!] Error: Need at least 2 valid images to combine.")
        sys.exit(1)
    
    # Display info if requested
    if args.info:
        print("\n[Image Information]")
        for idx, img in enumerate(images, 1):
            info = tool.get_image_info(img)
            print(f"\nImage {idx}: {os.path.basename(image_paths[idx-1])}")
            print(f"  Size: {info['width']}x{info['height']}")
            print(f"  Channels: {info['channels']}")
            print(f"  Mean RGB: ({info['mean_r']:.1f}, {info['mean_g']:.1f}, {info['mean_b']:.1f})")
            print(f"  Std Dev: {info['std']:.1f}")
    
    # Process
    result = None
    if args.mode == 'smart':
        result = tool.smart_stitch(images)
        if result is not None and not args.no_crop:
            tool.log("Auto-cropping empty borders...")
            result = tool.crop_black_borders(result)
    elif args.mode == 'grid':
        result = tool.grid_stitch(images, cols=args.grid_cols, spacing=args.grid_spacing,
                                 bg_color=tuple(args.grid_bg))
    else:
        result = tool.simple_stitch(images, direction=args.mode)
    
    if result is None:
        print("[!] Processing failed.")
        sys.exit(1)
    
    # Apply custom crop
    if args.crop:
        result = tool.crop_custom(result, *args.crop)
        tool.log("Applied custom crop")
    
    # Add watermark
    if args.watermark:
        result = tool.add_watermark(result, args.watermark, args.watermark_pos,
                                   args.watermark_opacity)
        tool.log(f"Added text watermark: {args.watermark}")
    
    if args.watermark_img:
        result = tool.add_image_watermark(result, args.watermark_img, args.watermark_pos,
                                         args.watermark_opacity, args.watermark_scale)
        tool.log(f"Added image watermark: {args.watermark_img}")
    
    # Determine output filename
    if args.output:
        out_name = args.output
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        ext = args.format or 'jpg'
        out_name = f"imgcom_result_{timestamp}.{ext}"
    
    # Ensure correct extension
    if args.format:
        base, _ = os.path.splitext(out_name)
        out_name = f"{base}.{args.format}"
    
    # Save
    if tool.save_image(result, out_name, quality=args.quality, 
                      preserve_exif=args.preserve_exif):
        tool.log(f"Image saved to: {out_name}", "SUCCESS")
        
        # Display final image info
        if args.verbose:
            info = tool.get_image_info(result)
            print(f"  Final size: {info['width']}x{info['height']}")
            print(f"  File size: {os.path.getsize(out_name) / 1024:.1f} KB")
        
        # macOS Preview
        if sys.platform == 'darwin' and not args.no_preview:
            os.system(f"open {out_name}")
    else:
        print("[!] Error: Failed to save image.")
        sys.exit(1)


if __name__ == "__main__":
    main()
