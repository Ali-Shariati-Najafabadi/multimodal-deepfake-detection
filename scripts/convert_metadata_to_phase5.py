#!/usr/bin/env python3
"""
Convert AV-Deepfake1M++ metadata to Phase 5 format with segment-level annotations.

Usage:
    python scripts/convert_metadata_to_phase5.py \
        --input /path/to/original_metadata.json \
        --output /path/to/converted_metadata.json
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List
from tqdm import tqdm


def convert_manipulation_format(manip: Dict, video_duration: float) -> Dict:
    """
    Convert manipulation entry to standardized format.
    
    Expected input formats:
    - {"start_frame": 75, "end_frame": 180, "fps": 25, "type": "face_swap"}
    - {"start_time": 2.5, "end_time": 6.0, "type": "audio_clone"}
    - {"start": 2.5, "end": 6.0, "manipulation_type": "deepfake"}
    """
    result = {}
    
    if 'start_time' in manip:
        result['start'] = float(manip['start_time'])
    elif 'start_frame' in manip and 'fps' in manip:
        result['start'] = float(manip['start_frame']) / float(manip['fps'])
    elif 'start' in manip:
        result['start'] = float(manip['start'])
    else:
        result['start'] = 0.0
    
    if 'end_time' in manip:
        result['end'] = float(manip['end_time'])
    elif 'end_frame' in manip and 'fps' in manip:
        result['end'] = float(manip['end_frame']) / float(manip['fps'])
    elif 'end' in manip:
        result['end'] = float(manip['end'])
    else:
        result['end'] = video_duration
    
    manip_type = manip.get('type') or manip.get('manipulation_type', 'unknown')
    
    if 'audio' in manip_type.lower():
        result['type'] = 'audio'
    elif 'video' in manip_type.lower() or 'face' in manip_type.lower():
        result['type'] = 'video'
    else:
        result['type'] = 'both'
    
    return result


def convert_metadata(input_file: str, output_file: str, verbose: bool = True):
    """
    Convert metadata to Phase 5 format.
    """
    print(f"\n{'='*60}")
    print("Converting Metadata to Phase 5 Format")
    print(f"{'='*60}\n")
    
    print(f"Loading: {input_file}")
    with open(input_file, 'r') as f:
        original_meta = json.load(f)
    
    print(f"Found {len(original_meta)} videos\n")
    
    converted_meta = {}
    
    real_count = 0
    fake_count = 0
    with_annotations = 0
    without_annotations = 0
    
    for video_id, info in tqdm(original_meta.items(), desc="Converting"):
        label = info.get('label', 0)
        if isinstance(label, str):
            label = 1 if label.lower() in ['fake', 'deepfake', '1'] else 0
        
        if label == 0:
            real_count += 1
        else:
            fake_count += 1
        
        duration = info.get('duration', 10.0)
        if 'num_frames' in info and 'fps' in info:
            duration = float(info['num_frames']) / float(info['fps'])
        
        converted_entry = {
            'path': info.get('path', ''),
            'label': int(label),
            'shard': info.get('shard', 0),
            'duration': float(duration),
            'manipulations': []
        }
        
        if 'fps' in info:
            converted_entry['fps'] = float(info['fps'])
        
        if label == 1:
            if 'manipulations' in info and info['manipulations']:
                for manip in info['manipulations']:
                    converted_manip = convert_manipulation_format(manip, duration)
                    converted_entry['manipulations'].append(converted_manip)
                with_annotations += 1
            elif 'segments' in info and info['segments']:
                for seg in info['segments']:
                    converted_manip = convert_manipulation_format(seg, duration)
                    converted_entry['manipulations'].append(converted_manip)
                with_annotations += 1
            else:
                converted_entry['manipulations'].append({
                    'start': 0.0,
                    'end': duration,
                    'type': 'both'
                })
                without_annotations += 1
        
        converted_meta[video_id] = converted_entry
    
    print(f"\nConversion Statistics:")
    print(f"  Total videos: {len(converted_meta)}")
    print(f"  Real videos: {real_count}")
    print(f"  Fake videos: {fake_count}")
    print(f"  Fake with segment annotations: {with_annotations}")
    print(f"  Fake without segment annotations: {without_annotations}")
    
    print(f"\nSaving to: {output_file}")
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump(converted_meta, f, indent=2)
    
    print(f"\n{'='*60}")
    print("Conversion Complete!")
    print(f"{'='*60}\n")
    
    if verbose:
        print("\nExample entry:")
        example_id = list(converted_meta.keys())[0]
        print(json.dumps({example_id: converted_meta[example_id]}, indent=2))


def validate_converted_metadata(metadata_file: str):
    """
    Validate converted metadata format.
    """
    print(f"\n{'='*60}")
    print("Validating Converted Metadata")
    print(f"{'='*60}\n")
    
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)
    
    errors = []
    warnings = []
    
    for video_id, info in metadata.items():
        if 'path' not in info:
            errors.append(f"{video_id}: Missing 'path' field")
        
        if 'label' not in info:
            errors.append(f"{video_id}: Missing 'label' field")
        
        if 'duration' not in info:
            warnings.append(f"{video_id}: Missing 'duration' field")
        
        if 'manipulations' not in info:
            errors.append(f"{video_id}: Missing 'manipulations' field")
        else:
            for i, manip in enumerate(info['manipulations']):
                if 'start' not in manip:
                    errors.append(f"{video_id}: Manipulation {i} missing 'start'")
                if 'end' not in manip:
                    errors.append(f"{video_id}: Manipulation {i} missing 'end'")
                if 'type' not in manip:
                    warnings.append(f"{video_id}: Manipulation {i} missing 'type'")
    
    if errors:
        print(f"❌ Found {len(errors)} errors:")
        for error in errors[:10]:
            print(f"  - {error}")
        if len(errors) > 10:
            print(f"  ... and {len(errors) - 10} more")
    else:
        print("✅ No errors found!")
    
    if warnings:
        print(f"\n⚠️  Found {len(warnings)} warnings:")
        for warning in warnings[:10]:
            print(f"  - {warning}")
        if len(warnings) > 10:
            print(f"  ... and {len(warnings) - 10} more")
    else:
        print("✅ No warnings!")
    
    print(f"\n{'='*60}\n")
    
    return len(errors) == 0


def main():
    parser = argparse.ArgumentParser(
        description='Convert AV-Deepfake1M++ metadata to Phase 5 format'
    )
    parser.add_argument('--input', type=str, required=True,
                       help='Path to original metadata.json')
    parser.add_argument('--output', type=str, required=True,
                       help='Path to save converted metadata.json')
    parser.add_argument('--validate', action='store_true',
                       help='Validate converted metadata after conversion')
    parser.add_argument('--verbose', action='store_true', default=True,
                       help='Print example entry')
    
    args = parser.parse_args()
    
    convert_metadata(args.input, args.output, args.verbose)
    
    if args.validate:
        is_valid = validate_converted_metadata(args.output)
        if not is_valid:
            print("⚠️  Validation failed! Please check errors above.")
            return 1
    
    return 0


if __name__ == '__main__':
    exit(main())
