"""
Visualize predictions on ATR-UMOD images.
Draws rotated bounding boxes on RGB/IR image pairs.
"""
import argparse
import sys
from pathlib import Path
import cv2
import numpy as np

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


CLASS_COLORS = {
    'car': (0, 255, 0),
    'suv': (255, 0, 0),
    'van': (0, 0, 255),
    'bus': (255, 255, 0),
    'freight_car': (255, 0, 255),
    'truck': (0, 255, 255),
    'motorcycle': (128, 128, 0),
    'trailer': (128, 0, 128),
    'tank_truck': (0, 128, 128),
    'excavator': (255, 128, 0),
    'crane': (128, 255, 0),
}


def draw_rotated_box(img, cx, cy, w, h, angle, color, thickness=2, label=None):
    """Draw a rotated bounding box on an image."""
    rect = ((cx, cy), (w, h), angle)
    box = cv2.boxPoints(rect)
    box = np.int32(box)
    cv2.drawContours(img, [box], 0, color, thickness)
    if label:
        cv2.putText(img, label, (int(cx - w / 2), int(cy - h / 2 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)


def visualize_pair(rgb_path, ir_path, label_path, out_path=None):
    """Show RGB and IR side-by-side with ground truth boxes."""
    import xml.etree.ElementTree as ET

    rgb = cv2.imread(str(rgb_path))
    ir = cv2.imread(str(ir_path))
    if rgb is None or ir is None:
        print(f'Cannot read images: {rgb_path}, {ir_path}')
        return

    ir_color = cv2.cvtColor(ir, cv2.COLOR_GRAY2BGR)

    if label_path and Path(label_path).exists():
        tree = ET.parse(str(label_path))
        for obj in tree.findall('object'):
            name = obj.find('name').text
            robndbox = obj.find('robndbox')
            cx = float(robndbox.find('cx').text)
            cy = float(robndbox.find('cy').text)
            w = float(robndbox.find('w').text)
            h = float(robndbox.find('h').text)
            angle = float(robndbox.find('angle').text)
            color = CLASS_COLORS.get(name, (255, 255, 255))
            draw_rotated_box(rgb, cx, cy, w, h, angle, color, label=name)
            draw_rotated_box(ir_color, cx, cy, w, h, angle, color, label=name)

    combined = np.hstack([rgb, ir_color])
    cv2.putText(combined, 'RGB', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(combined, 'IR', (rgb.shape[1] + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    if out_path:
        cv2.imwrite(str(out_path), combined)
    else:
        cv2.imshow('ATR-UMOD', combined)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description='Visualize ATR-UMOD samples')
    parser.add_argument('--split', default='data/train', help='data/train or data/val')
    parser.add_argument('--index', type=int, default=0, help='sample index to show')
    parser.add_argument('--out', help='output image path')
    args = parser.parse_args()

    data_root = project_root / args.split
    image_files = sorted((data_root / 'images').glob('*.jpg'))
    if args.index >= len(image_files):
        print(f'Index {args.index} out of range (max {len(image_files)-1})')
        return

    img_path = image_files[args.index]
    basename = img_path.stem
    ir_path = data_root / 'images_ir' / f'{basename}.jpg'
    label_path = data_root / 'labels' / f'{basename}.xml'

    visualize_pair(img_path, ir_path, label_path, args.out)


if __name__ == '__main__':
    main()
