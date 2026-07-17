import cv2
import numpy as np
import h5py
import json
from pathlib import Path
from tqdm import tqdm
import multiprocessing as mp
from typing import List, Tuple, Optional
import urllib.request
import warnings
import av
import librosa
warnings.filterwarnings('ignore')


_DETECTOR       = None
_FACE_SIZE      = 224
_NUM_FRAMES     = 16
_AUDIO_SR       = 16000
_AUDIO_DURATION = 4.0
_CONFIDENCE     = 0.7


FAKEAVCELEB_4CLASS = {
    'RealVideo-RealAudio': 0,
    'RealVideo-FakeAudio': 1,
    'FakeVideo-RealAudio': 2,
    'FakeVideo-FakeAudio': 3,
}
FAKEAVCELEB_BINARY = {
    'RealVideo-RealAudio': 0,
    'RealVideo-FakeAudio': 1,
    'FakeVideo-RealAudio': 1,
    'FakeVideo-FakeAudio': 1,
}
AVDEEPFAKE_BINARY = {
    'real':            0,
    'audio_modified':  1,
    'visual_modified': 1,
    'both_modified':   1,
}
AVDEEPFAKE_4CLASS = {
    'real':            0,
    'audio_modified':  1,
    'visual_modified': 2,
    'both_modified':   3,
}


def compute_temporal_indices(
    total_frames:          int,
    fps:                   float,
    visual_fake_segments,
    audio_fake_segments,
    modify_type:           str,
    num_frames:            int,
    fake_ratio:            float = 0.75
) -> np.ndarray:
    if modify_type == 'real' or total_frames < num_frames:
        return np.linspace(0, total_frames - 1, num_frames, dtype=int)

    fake_frame_set = set()

    if modify_type in ('visual_modified', 'both_modified') and visual_fake_segments:
        for seg in visual_fake_segments:
            try:
                s = max(0, int(float(seg[0]) * fps))
                e = min(total_frames, max(s + 1, int(float(seg[1]) * fps)))
                fake_frame_set.update(range(s, e))
            except (TypeError, ValueError, IndexError):
                pass

    if modify_type in ('audio_modified', 'both_modified') and audio_fake_segments:
        for seg in audio_fake_segments:
            try:
                s = max(0, int((float(seg[0]) - 0.5) * fps))
                e = min(total_frames, max(s + 1, int((float(seg[1]) + 0.5) * fps)))
                fake_frame_set.update(range(s, e))
            except (TypeError, ValueError, IndexError):
                pass

    all_fake_frames = sorted(fake_frame_set)

    if len(all_fake_frames) < 4:
        all_segs = (visual_fake_segments or []) + (audio_fake_segments or [])
        for seg in all_segs:
            try:
                mid = (float(seg[0]) + float(seg[1])) / 2.0
                s   = max(0, int((mid - 1.0) * fps))
                e   = min(total_frames, int((mid + 1.0) * fps))
                fake_frame_set.update(range(s, e))
            except (TypeError, ValueError, IndexError):
                pass
        all_fake_frames = sorted(fake_frame_set)

    if not all_fake_frames:
        return np.linspace(0, total_frames - 1, num_frames, dtype=int)

    n_fake = max(1, int(num_frames * fake_ratio))
    n_rest = num_frames - n_fake

    if len(all_fake_frames) >= n_fake:
        step = max(1, len(all_fake_frames) // n_fake)
        fake_indices = all_fake_frames[::step][:n_fake]
    else:
        fake_indices = all_fake_frames
        n_rest = num_frames - len(fake_indices)

    rest_indices = []
    if n_rest > 0:
        fake_set  = set(fake_indices)
        rest_pool = [i for i in range(total_frames) if i not in fake_set]
        if rest_pool:
            step = max(1, len(rest_pool) // (n_rest * 2))
            rest_indices = rest_pool[::step][:n_rest]

    combined = sorted(set(list(fake_indices) + list(rest_indices)))

    if len(combined) < num_frames:
        extra = list(np.linspace(0, total_frames - 1,
                                 num_frames - len(combined), dtype=int))
        combined = sorted(set(combined + extra))

    return np.array(combined[:num_frames], dtype=int)


def compute_audio_segment_bounds(
    total_samples:        int,
    audio_sr:             int,
    visual_fake_segments,
    audio_fake_segments,
    modify_type:          str,
    target_samples:       int
) -> Tuple[int, int]:
    def center_of_segment(segs):
        try:
            start_sec = float(segs[0][0])
            end_sec   = float(segs[0][1])
            return (start_sec + end_sec) / 2.0
        except (TypeError, ValueError, IndexError):
            return None

    mid_sec = None

    if modify_type in ('audio_modified', 'both_modified') and audio_fake_segments:
        mid_sec = center_of_segment(audio_fake_segments)
    elif modify_type == 'visual_modified' and visual_fake_segments:
        mid_sec = center_of_segment(visual_fake_segments)

    if mid_sec is not None:
        mid_sample = int(mid_sec * audio_sr)
    else:
        mid_sample = total_samples // 2

    start = max(0, mid_sample - target_samples // 2)
    end   = start + target_samples
    if end > total_samples:
        start = max(0, total_samples - target_samples)
        end   = total_samples
    return start, end


# =====================================================================
# OPENCV FACE DETECTOR
# =====================================================================
def download_opencv_model(model_dir: Path):
    model_dir.mkdir(parents=True, exist_ok=True)
    prototxt   = model_dir / "deploy.prototxt"
    caffemodel = model_dir / "res10_300x300_ssd_iter_140000.caffemodel"
    if not prototxt.exists():
        print("Downloading deploy.prototxt...")
        urllib.request.urlretrieve(
            "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt",
            prototxt)
    if not caffemodel.exists():
        print("Downloading caffemodel (10MB)...")
        urllib.request.urlretrieve(
            "https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel",
            caffemodel)
    return str(prototxt), str(caffemodel)


def init_worker(face_size, num_frames, audio_sr, audio_duration, model_dir):
    global _DETECTOR, _FACE_SIZE, _NUM_FRAMES, _AUDIO_SR, _AUDIO_DURATION
    prototxt, caffemodel = download_opencv_model(Path(model_dir))
    _DETECTOR       = cv2.dnn.readNetFromCaffe(prototxt, caffemodel)
    _FACE_SIZE      = face_size
    _NUM_FRAMES     = num_frames
    _AUDIO_SR       = audio_sr
    _AUDIO_DURATION = audio_duration


def detect_face_opencv(frame_rgb: np.ndarray) -> Optional[Tuple[int,int,int,int]]:
    global _DETECTOR, _CONFIDENCE
    h, w = frame_rgb.shape[:2]
    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    blob = cv2.dnn.blobFromImage(frame_bgr, 1.0, (300,300), (104.,177.,123.))
    _DETECTOR.setInput(blob)
    detections = _DETECTOR.forward()
    best_box, best_conf = None, 0
    for i in range(detections.shape[2]):
        conf = detections[0, 0, i, 2]
        if conf > _CONFIDENCE and conf > best_conf:
            best_conf = conf
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            best_box = tuple(box.astype(int))
    return best_box


# =====================================================================
# VIDEO EXTRACTION
# =====================================================================
def _extract_frames_from_indices(
    video_path: str,
    indices:    np.ndarray,
    face_size:  int
) -> Optional[np.ndarray]:
    """Liest Frames an den angegebenen Indizes und extrahiert Gesichter."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    indices      = np.clip(indices, 0, max(0, total_frames - 1))

    mean  = np.array([0.485, 0.456, 0.406])
    std   = np.array([0.229, 0.224, 0.225])
    faces = []

    for target_idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(target_idx))
        ret, frame = cap.read()
        if not ret:
            faces.append(np.zeros((3, face_size, face_size), dtype=np.float32))
            continue

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        bbox      = detect_face_opencv(frame_rgb)

        if bbox is not None:
            x1, y1, x2, y2 = bbox
            h, w = frame_rgb.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            if x2 > x1 and y2 > y1:
                face = frame_rgb[y1:y2, x1:x2]
                if face.size > 0:
                    face = cv2.resize(face, (face_size, face_size))
                    face = face.astype(np.float32) / 255.0
                    face = (face - mean) / std
                    face = face.transpose(2, 0, 1)
                    faces.append(face)
                    continue
        faces.append(np.zeros((3, face_size, face_size), dtype=np.float32))

    cap.release()

    num_frames = len(indices)
    if sum(1 for f in faces if f.sum() != 0) < num_frames // 2:
        return None

    return np.stack(faces[:num_frames]).astype(np.float32)


def extract_uniform_frames(video_path, num_frames, face_size):
    """FakeAVCeleb / DeepSpeak / AV-Deepfake (uniform): uniform sampling."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    if total < num_frames:
        return None
    indices = np.linspace(0, total - 1, num_frames, dtype=int)
    return _extract_frames_from_indices(video_path, indices, face_size)


def extract_temporal_frames(
    video_path:            str,
    num_frames:            int,
    face_size:             int,
    visual_fake_segments,
    audio_fake_segments,
    modify_type:           str
) -> Optional[np.ndarray]:
    """AV-Deepfake (nur --segment-aware): temporal-aware sampling mit Bias zum Fake-Segment."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps          = cap.get(cv2.CAP_PROP_FPS) or 25.0
    cap.release()
    if total_frames < num_frames:
        return None

    indices = compute_temporal_indices(
        total_frames, fps,
        visual_fake_segments,
        audio_fake_segments,
        modify_type, num_frames
    )
    return _extract_frames_from_indices(video_path, indices, face_size)


# =====================================================================
# AUDIO EXTRACTION
# =====================================================================
def _load_full_audio(video_path: str, audio_sr: int):
    """Lädt komplettes Audio aus Video-Datei."""
    container = av.open(str(video_path))
    if len(container.streams.audio) == 0:
        container.close()
        return None, audio_sr
    orig_sr    = container.streams.audio[0].rate
    audio_data = []
    for frame in container.decode(audio=0):
        audio_data.append(frame.to_ndarray())
    container.close()
    if not audio_data:
        return None, orig_sr
    audio = np.concatenate(audio_data, axis=1)[0].astype(np.float32)
    if orig_sr != audio_sr:
        audio = librosa.resample(audio, orig_sr=orig_sr, target_sr=audio_sr)
    return audio, audio_sr


def _normalize_audio(segment: np.ndarray, target_samples: int) -> np.ndarray:
    """Normalisiert und padded/trimmt auf target_samples."""
    if len(segment) > target_samples:
        segment = segment[:target_samples]
    elif len(segment) < target_samples:
        segment = np.pad(segment, (0, target_samples - len(segment)))
    rms = np.sqrt(np.mean(segment**2))
    if rms > 0:
        segment = segment / (rms * 10)
    return np.clip(segment, -1.0, 1.0).astype(np.float32)


def extract_center_audio(video_path, audio_sr, audio_duration):
    """FakeAVCeleb / DeepSpeak / AV-Deepfake (uniform): Audio aus Mitte."""
    target_samples = int(audio_sr * audio_duration)
    try:
        audio, _ = _load_full_audio(video_path, audio_sr)
        if audio is None:
            return np.zeros(target_samples, dtype=np.float32)
        center = len(audio) // 2
        start  = max(0, center - target_samples // 2)
        return _normalize_audio(audio[start:], target_samples)
    except Exception:
        return np.zeros(target_samples, dtype=np.float32)


def extract_temporal_audio(
    video_path:            str,
    audio_sr:              int,
    audio_duration:        float,
    visual_fake_segments,
    audio_fake_segments,
    modify_type:           str
) -> np.ndarray:
    """AV-Deepfake (nur --segment-aware): Audio zentriert auf manipuliertes Segment."""
    target_samples = int(audio_sr * audio_duration)
    try:
        audio, _ = _load_full_audio(video_path, audio_sr)
        if audio is None:
            return np.zeros(target_samples, dtype=np.float32)
        start, end = compute_audio_segment_bounds(
            len(audio), audio_sr,
            visual_fake_segments,
            audio_fake_segments,
            modify_type, target_samples
        )
        return _normalize_audio(audio[start:end], target_samples)
    except Exception:
        return np.zeros(target_samples, dtype=np.float32)


# =====================================================================
# WORKERS
# =====================================================================
def process_single_video(args) -> Optional[dict]:
    """FakeAVCeleb / DeepSpeak / AV-Deepfake (uniform) — uniform sampling."""
    global _FACE_SIZE, _NUM_FRAMES, _AUDIO_SR, _AUDIO_DURATION
    video_path, label, video_id = args
    try:
        faces = extract_uniform_frames(video_path, _NUM_FRAMES, _FACE_SIZE)
        if faces is None:
            return None
        audio = extract_center_audio(video_path, _AUDIO_SR, _AUDIO_DURATION)
        return {
            'video_id':   video_id,
            'path':       str(video_path),
            'label':      int(label),
            'faces':      faces,
            'audio':      audio,
            'num_frames': len(faces)
        }
    except Exception:
        return None


def process_single_video_temporal(args) -> Optional[dict]:
    """
    AV-Deepfake — temporal-aware sampling (nur --segment-aware).
    args: (video_path, label, video_id, modify_type,
           visual_fake_segments, audio_fake_segments)
    """
    global _FACE_SIZE, _NUM_FRAMES, _AUDIO_SR, _AUDIO_DURATION
    (video_path, label, video_id,
     modify_type, visual_fake_segments, audio_fake_segments) = args
    try:
        faces = extract_temporal_frames(
            video_path, _NUM_FRAMES, _FACE_SIZE,
            visual_fake_segments, audio_fake_segments, modify_type
        )
        if faces is None:
            return None
        audio = extract_temporal_audio(
            video_path, _AUDIO_SR, _AUDIO_DURATION,
            visual_fake_segments, audio_fake_segments, modify_type
        )
        return {
            'video_id':    video_id,
            'path':        str(video_path),
            'label':       int(label),
            'modify_type': modify_type,
            'faces':       faces,
            'audio':       audio,
            'num_frames':  len(faces)
        }
    except Exception:
        return None


# =====================================================================
# SCAN FAKEAVCELEB
# =====================================================================
def scan_fakeavceleb(
    dataset_path: str,
    binary:       bool  = True,
    train_ratio:  float = 0.70,
    val_ratio:    float = 0.15,
    seed:         int   = 42
):
    dataset_path = Path(dataset_path)
    label_map    = FAKEAVCELEB_BINARY if binary else FAKEAVCELEB_4CLASS

    print(f"\nScanning FakeAVCeleb v1.2: {dataset_path}")
    print(f"Mode: {'Binary' if binary else '4-Klassen'}")

    all_videos = []
    for class_folder, label in label_map.items():
        class_path = dataset_path / class_folder
        if not class_path.exists():
            print(f"  WARNING: {class_folder} nicht gefunden!")
            continue
        videos = list(class_path.rglob("*.mp4"))
        for vid_path in videos:
            identity_id = None
            for part in vid_path.parts:
                if part.startswith('id') and len(part) == 7:
                    identity_id = part
                    break
            all_videos.append({
                'path':     str(vid_path),
                'label':    label,
                'class':    class_folder,
                'identity': identity_id or 'unknown'
            })
        print(f"  {class_folder}: {len(videos):,} Videos → Label {label}")

    print(f"\nGesamt: {len(all_videos):,} Videos")

    import random
    random.seed(seed)
    identities = list(set(v['identity'] for v in all_videos))
    random.shuffle(identities)
    n       = len(identities)
    n_train = int(n * train_ratio)
    n_val   = int(n * val_ratio)
    train_ids = set(identities[:n_train])
    val_ids   = set(identities[n_train:n_train + n_val])
    test_ids  = set(identities[n_train + n_val:])

    def make_split(id_set):
        vids = [v for v in all_videos if v['identity'] in id_set]
        return {
            'video_list': [v['path']  for v in vids],
            'labels':     [v['label'] for v in vids],
        }

    train_data = make_split(train_ids)
    val_data   = make_split(val_ids)
    test_data  = make_split(test_ids)

    for name, data in [('Train', train_data), ('Val', val_data), ('Test', test_data)]:
        labels = data['labels']
        print(f"\n{name}: {len(labels):,} Videos")
        if binary:
            print(f"  Real (0): {labels.count(0):,}  |  Fake (1): {labels.count(1):,}")
        else:
            for i, cn in enumerate(FAKEAVCELEB_4CLASS.keys()):
                print(f"  Klasse {i} ({cn}): {labels.count(i):,}")

    return train_data, val_data, test_data


# =====================================================================
# SHARDED CACHER
# =====================================================================
class ShardedVideoCacher:
    def __init__(
        self,
        cache_dir:      str,
        face_size:      int   = 224,
        num_frames:     int   = 16,
        audio_sr:       int   = 16000,
        audio_duration: float = 4.0,
        shard_size:     int   = 5000,
        num_workers:    int   = 32,
        cache_audio:    bool  = True
    ):
        self.cache_dir      = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.face_size      = face_size
        self.num_frames     = num_frames
        self.audio_sr       = audio_sr
        self.audio_duration = audio_duration
        self.shard_size     = shard_size
        self.num_workers    = num_workers
        self.cache_audio    = cache_audio
        self.model_dir      = self.cache_dir / "opencv_model"
        download_opencv_model(self.model_dir)

        print(f"\nSharded Cacher initialisiert:")
        print(f"  Cache dir:        {cache_dir}")
        print(f"  Frames/Video:     {num_frames}")
        print(f"  Frame Size:       {face_size}x{face_size}")
        print(f"  Audio SR:         {audio_sr} Hz")
        print(f"  Audio Segment:    {audio_duration} Sek.")
        print(f"  Shard Size:       {shard_size}")
        print(f"  Workers:          {num_workers}")

    def cache_dataset(self, dataset_name, split, video_list, labels,
                      resume_from=0, temporal_args=None):
        """
        temporal_args: Liste von 6-Tupeln für AV-Deepfake (--segment-aware),
                       None für uniform (FakeAVCeleb/DeepSpeak/AV-Deepfake default)
        """
        total_videos = len(video_list)
        num_shards   = (total_videos + self.shard_size - 1) // self.shard_size

        print(f"\n{'='*70}")
        print(f"Caching {dataset_name.upper()} {split.upper()}")
        mode_str = "temporal-aware" if temporal_args else "uniform"
        print(f"Sampling: {mode_str}")
        print(f"{'='*70}")
        print(f"  Total: {total_videos:,}  |  Shards: {num_shards}")

        all_metadata = {}
        meta_file    = self.cache_dir / f"{dataset_name}_{split}_metadata.json"

        if resume_from > 0 and meta_file.exists():
            with open(meta_file) as f:
                all_metadata = json.load(f)
            print(f"  Resume von {resume_from:,}, {len(all_metadata):,} bereits gecacht")

        start_shard = resume_from // self.shard_size
        worker_fn   = process_single_video_temporal if temporal_args else process_single_video

        for shard_idx in range(start_shard, num_shards):
            start_idx  = shard_idx * self.shard_size
            end_idx    = min(start_idx + self.shard_size, total_videos)
            shard_file = self.cache_dir / f"{dataset_name}_{split}_shard{shard_idx:04d}.h5"

            if shard_file.exists() and shard_idx < start_shard:
                continue

            print(f"\n  SHARD {shard_idx+1}/{num_shards} "
                  f"(Videos {start_idx:,}–{end_idx:,})")

            if temporal_args:
                batch_args = temporal_args[start_idx:end_idx]
            else:
                batch_args = [
                    (video_list[i], labels[i], f"video_{i:08d}")
                    for i in range(start_idx, end_idx)
                ]

            successful = failed = 0

            with mp.Pool(
                processes=self.num_workers,
                initializer=init_worker,
                initargs=(self.face_size, self.num_frames,
                          self.audio_sr, self.audio_duration, str(self.model_dir))
            ) as pool:
                results = list(tqdm(
                    pool.imap(worker_fn, batch_args),
                    total=len(batch_args),
                    desc=f"Shard {shard_idx+1}"
                ))

            shard_metadata = {}
            with h5py.File(shard_file, 'w') as f:
                f.attrs['num_frames']     = self.num_frames
                f.attrs['face_size']      = self.face_size
                f.attrs['audio_sr']       = self.audio_sr
                f.attrs['audio_duration'] = self.audio_duration
                f.attrs['temporal_aware'] = temporal_args is not None

                for result in results:
                    if result is None:
                        failed += 1
                        continue
                    vid = result['video_id']
                    f.create_dataset(f"{vid}/faces", data=result['faces'],
                                     compression='lzf')
                    if self.cache_audio:
                        f.create_dataset(f"{vid}/audio", data=result['audio'],
                                         compression='lzf')
                    shard_metadata[vid] = {
                        'shard':      shard_idx,
                        'path':       result['path'],
                        'label':      result['label'],
                        'num_frames': result['num_frames'],
                    }
                    if 'modify_type' in result:
                        shard_metadata[vid]['modify_type'] = result['modify_type']
                    successful += 1

            all_metadata.update(shard_metadata)
            with open(meta_file, 'w') as f:
                json.dump(all_metadata, f)

            gb = shard_file.stat().st_size / (1024**3)
            print(f"  ✓ {successful:,} OK | {failed:,} failed | {gb:.2f} GB")

        index_file = self.cache_dir / f"{dataset_name}_{split}_index.json"
        with open(index_file, 'w') as f:
            json.dump({
                'dataset':        dataset_name,
                'split':          split,
                'total_videos':   len(all_metadata),
                'num_shards':     num_shards,
                'num_frames':     self.num_frames,
                'face_size':      self.face_size,
                'audio_sr':       self.audio_sr,
                'audio_duration': self.audio_duration,
                'temporal_aware': temporal_args is not None,
                'shards': [
                    str(self.cache_dir / f"{dataset_name}_{split}_shard{i:04d}.h5")
                    for i in range(num_shards)
                ]
            }, f, indent=2)

        all_labels = [m['label'] for m in all_metadata.values()]
        total_gb   = sum(
            (self.cache_dir / f"{dataset_name}_{split}_shard{i:04d}.h5").stat().st_size
            for i in range(num_shards)
            if (self.cache_dir / f"{dataset_name}_{split}_shard{i:04d}.h5").exists()
        ) / (1024**3)

        print(f"\n{'='*70}")
        print(f"FERTIG: {dataset_name.upper()} {split.upper()}")
        print(f"  Gesamt gecacht: {len(all_metadata):,} Videos")
        unique_labels = sorted(set(all_labels))
        if not unique_labels or max(unique_labels) <= 1:
            print(f"  Real (0): {all_labels.count(0):,} | Fake (1): {all_labels.count(1):,}")
        else:
            label_names = {0: 'Real', 1: 'AudioMod', 2: 'VisualMod', 3: 'BothMod'}
            for lbl in unique_labels:
                print(f"  Klasse {lbl} ({label_names.get(lbl,'?')}): {all_labels.count(lbl):,}")
        print(f"  Gesamtgröße: {total_gb:.2f} GB")
        print(f"{'='*70}\n")



def cache_av_deepfake(
    cacher,
    dataset_path,
    split,
    max_videos,
    resume_from,
    binary:        bool = True,
    real_only:     bool = False,
    segment_aware: bool = False
):
    import random
    random.seed(42)

    dataset_path = Path(dataset_path)
    fake_types   = ['audio_modified', 'visual_modified', 'both_modified']

    with open(dataset_path / f"{split}_metadata.json") as f:
        raw_metadata = json.load(f)

    print(f"\n  Gesamte {split}-Metadata: {len(raw_metadata):,} Einträge")
    print(f"  Sampling: {'temporal-aware (Oracle)' if segment_aware else 'uniform'}")

    # ------------------------------------------------------------------
    # SAMPLING (Auswahl der Videos + Label-Zuordnung)
    # ------------------------------------------------------------------
    if real_only:
        metadata = [s for s in raw_metadata if s['modify_type'] == 'real']
        print(f"  [real_only] {len(metadata):,} Real-Samples")
        if max_videos and len(metadata) > max_videos:
            random.shuffle(metadata)
            metadata = metadata[:max_videos]
        label_fn = lambda s: 0

    elif binary:
        half     = (max_videos // 2) if max_videos else None
        per_fake = (half // 3)       if half       else None

        real_items = [s for s in raw_metadata if s['modify_type'] == 'real']
        random.shuffle(real_items)
        sampled = real_items[:half] if half else real_items

        for ft in fake_types:
            items = [s for s in raw_metadata if s['modify_type'] == ft]
            random.shuffle(items)
            sampled.extend(items[:per_fake] if per_fake else items)

        metadata   = sampled
        real_count = sum(1 for s in metadata if s['modify_type'] == 'real')
        fake_count = len(metadata) - real_count
        print(f"  Balanced 1:1: {real_count:,} Real + {fake_count:,} Fake = {len(metadata):,}")
        for ft in fake_types:
            print(f"    └ {ft}: {sum(1 for s in metadata if s['modify_type']==ft):,}")
        label_fn = lambda s: AVDEEPFAKE_BINARY.get(s['modify_type'], 0)

    else:  # 4class
        per_class = (max_videos // 4) if max_videos else None
        sampled   = []
        for t in ['real'] + fake_types:
            items = [s for s in raw_metadata if s['modify_type'] == t]
            random.shuffle(items)
            sampled.extend(items[:per_class] if per_class else items)

        metadata = sampled
        print(f"  4-Klassen: {len(metadata):,}")
        label_names_list = ['Real', 'AudioMod', 'VisualMod', 'BothMod']
        for i, t in enumerate(['real'] + fake_types):
            print(f"  Klasse {i} ({label_names_list[i]}): {sum(1 for s in metadata if s['modify_type']==t):,}")
        label_fn = lambda s: AVDEEPFAKE_4CLASS.get(s['modify_type'], 0)

    
    video_list    = []
    labels        = []
    temporal_args = [] if segment_aware else None

    for i, s in enumerate(metadata):
        video_path = str(dataset_path / split / split / s['file'])
        label      = label_fn(s)
        video_id   = f"video_{i:08d}"

        video_list.append(video_path)
        labels.append(label)

        if segment_aware:
            temporal_args.append((
                video_path, label, video_id,
                s.get('modify_type', 'real'),
                s.get('visual_fake_segments', []) or [],
                s.get('audio_fake_segments',  []) or []
            ))

    cacher.cache_dataset(
        'av_deepfake', split,
        video_list, labels,
        resume_from,
        temporal_args=temporal_args
    )


# =====================================================================
# DEEPSPEAK
# =====================================================================
def cache_deepspeak(cacher, dataset_path, split, max_videos, resume_from):
    import pandas as pd
    dataset_path = Path(dataset_path)

    split_def = pd.read_csv(dataset_path / "annotations-split-def.csv")
    df_fake   = pd.read_csv(dataset_path / "annotations-fake.csv")
    df_real   = pd.read_csv(dataset_path / "annotations-real.csv")

    split_key = 'train' if split == 'train' else 'test'
    ids       = split_def[split_def["split"] == split_key]["identity"].to_list()

    fake_df = df_fake[
        (df_fake["identity-source"].isin(ids)) &
        (df_fake["identity-target"].isin(ids))
    ]
    real_df = df_real[df_real["identity"].isin(ids)]

    video_list, labels = [], []
    for _, row in fake_df.iterrows():
        video_list.append(str(dataset_path / row["engine"] / row["video-file"]))
        labels.append(1)
    for _, row in real_df.iterrows():
        video_list.append(str(dataset_path / "real" / row["video-file"]))
        labels.append(0)

    if max_videos:
        video_list = video_list[:max_videos]
        labels     = labels[:max_videos]

    print(f"  Real: {labels.count(0):,} | Fake: {labels.count(1):,}")
    cacher.cache_dataset('deepspeak', split, video_list, labels, resume_from)


# =====================================================================
# MAIN
# =====================================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Deepfake Dataset Preprocessor v3 (uniform default; AV-Deepfake optional segment-aware)")
    parser.add_argument('--dataset',
                        choices=['fakeavceleb', 'av_deepfake', 'deepspeak', 'all'],
                        required=True)
    parser.add_argument('--split',
                        choices=['train', 'val', 'test', 'all'],
                        default='all')
    parser.add_argument('--mode',
                        choices=['binary', '4class'],
                        default='binary')
    parser.add_argument('--real-only',     action='store_true')
    parser.add_argument('--segment-aware', action='store_true',
                        help='AV-Deepfake: altes Oracle-Sampling (nur Vergleich); Default ist uniform')
    parser.add_argument('--cache-dir',
                        default='/data/shared/ali/deepfake-detection/cache')
    parser.add_argument('--num-workers',    type=int,   default=32)
    parser.add_argument('--shard-size',     type=int,   default=5000)
    parser.add_argument('--num-frames',     type=int,   default=16)
    parser.add_argument('--audio-duration', type=float, default=4.0)
    parser.add_argument('--max-videos',     type=int,   default=None)
    parser.add_argument('--no-audio',       action='store_true')
    parser.add_argument('--resume-from',    type=int,   default=0)
    args = parser.parse_args()

    cacher = ShardedVideoCacher(
        cache_dir=args.cache_dir,
        face_size=224,
        num_frames=args.num_frames,
        audio_sr=16000,
        audio_duration=args.audio_duration,
        shard_size=args.shard_size,
        num_workers=args.num_workers,
        cache_audio=not args.no_audio
    )

    if args.dataset in ['fakeavceleb', 'all']:
        train_data, val_data, test_data = scan_fakeavceleb(
            '/data/shared/Datasets/FakeAVCeleb_v1.2',
            binary=(args.mode == 'binary')
        )
        split_map = {'train': train_data, 'val': val_data, 'test': test_data}
        for split_name in (['train','val','test'] if args.split == 'all' else [args.split]):
            d = split_map[split_name]
            cacher.cache_dataset(
                'fakeavceleb', split_name,
                d['video_list'], d['labels'],
                args.resume_from if split_name == 'train' else 0
            )

    if args.dataset in ['av_deepfake', 'all']:
        for split_name in (['train','val'] if args.split == 'all' else [args.split]):
            print(f"\n{'='*70}\nAV-DEEPFAKE {split_name.upper()}\n{'='*70}")
            cache_av_deepfake(
                cacher,
                '/data/shared/Datasets/AV-Deepfake1M-PlusPlus',
                split_name,
                args.max_videos,
                args.resume_from if split_name == 'train' else 0,
                binary=(args.mode == 'binary'),
                real_only=args.real_only,
                segment_aware=args.segment_aware
            )

    if args.dataset in ['deepspeak', 'all']:
        for split_name in (['train','val'] if args.split == 'all' else [args.split]):
            print(f"\n{'='*70}\nDEEPSPEAK {split_name.upper()}\n{'='*70}")
            cache_deepspeak(
                cacher,
                '/data/shared/Datasets/DeepspeakV2',
                split_name,
                args.max_videos,
                args.resume_from if split_name == 'train' else 0
            )

    print("\n" + "="*70)
    print("ALLES FERTIG!")
    print("="*70)