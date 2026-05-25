"""
Convert ATR-UMOD XML labels to DOTA format for MMRotate.
DOTA format: x1 y1 x2 y2 x3 y3 x4 y4 classname difficult
One .txt file per image, same basename.
"""
import os
import xml.etree.ElementTree as ET
from pathlib import Path
from tqdm import tqdm

CLASS_NAMES = [
    'car', 'suv', 'van', 'bus', 'freight_car', 'truck',
    'motorcycle', 'trailer', 'tank_truck', 'excavator', 'crane'
]


def xml_to_dota(xml_path, txt_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    lines = []
    for obj in root.findall('object'):
        name = obj.find('name').text
        if name not in CLASS_NAMES:
            continue
        poly = obj.find('polygon')
        x1 = poly.find('x1').text
        y1 = poly.find('y1').text
        x2 = poly.find('x2').text
        y2 = poly.find('y2').text
        x3 = poly.find('x3').text
        y3 = poly.find('y3').text
        x4 = poly.find('x4').text
        y4 = poly.find('y4').text
        difficult = obj.find('difficult').text if obj.find('difficult') is not None else '0'
        lines.append(f'{x1} {y1} {x2} {y2} {x3} {y3} {x4} {y4} {name} {difficult}')
    with open(txt_path, 'w') as f:
        f.write('\n'.join(lines))


def convert_split(split_dir):
    """Convert all XML labels in a split to DOTA txt files."""
    for subset in ['labels', 'labels_ir']:
        xml_dir = Path(split_dir) / subset
        out_dir = Path(split_dir) / 'dota_labels' if subset == 'labels' else Path(split_dir) / 'dota_labels_ir'
        out_dir.mkdir(parents=True, exist_ok=True)
        xml_files = sorted(xml_dir.glob('*.xml'))
        for xml_path in tqdm(xml_files, desc=f'Converting {split_dir}/{subset}'):
            txt_path = out_dir / f'{xml_path.stem}.txt'
            xml_to_dota(xml_path, txt_path)
        print(f'  -> {len(xml_files)} files to {out_dir}')


if __name__ == '__main__':
    base = Path(__file__).parent.parent
    for split in ['data/train', 'data/val']:
        split_path = base / split
        if split_path.exists():
            convert_split(str(split_path))
        else:
            print(f'Skipping {split_path} (not found)')
    print('Done.')
