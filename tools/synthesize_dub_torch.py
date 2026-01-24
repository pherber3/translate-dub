#!/usr/bin/env python3
"""Synthesize dubbed audio from translated transcript using Qwen3-TTS (PyTorch).

Takes a translated transcript JSON and speaker clips metadata, then generates
speech for each segment using Qwen3-TTS voice cloning with the appropriate
speaker's voice.

Output: individual segment WAVs + combined audio file.
"""

import argparse
import json
import os
import time
from collections import defaultdict
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch

# Enable flash attention on ROCm/gfx1100
os.environ.setdefault("TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL", "1")


def trim_silence(audio: np.ndarray, sr: int, top_db: int = 25) -> np.ndarray:
    """Trim leading and trailing silence from audio."""
    trimmed, _ = librosa.effects.trim(audio, top_db=top_db)
    return trimmed


def load_speaker_references(clips_metadata_path: Path) -> dict:
    """Load speaker reference clips from metadata, using rank-1 clip per speaker.

    Returns dict mapping speaker_id -> {file, text, duration, rank}
    """
    with open(clips_metadata_path) as f:
        metadata = json.load(f)

    references = {}
    for clip in metadata['clips']:
        speaker = clip['speaker']
        # Only keep the best (rank 1) clip per speaker
        if speaker not in references or clip['rank'] < references[speaker]['rank']:
            references[speaker] = {
                'file': clip['file'],
                'text': clip['text'],
                'duration': clip['duration'],
                'rank': clip['rank'],
            }

    return references


def main():
    parser = argparse.ArgumentParser(
        description="Synthesize dubbed audio from translated transcript (PyTorch/Qwen3-TTS)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python synthesize_dub_torch.py transcript_en.json

  # Specify clips metadata explicitly
  python synthesize_dub_torch.py transcript_en.json --clips clips_metadata.json

  # Custom output directory
  python synthesize_dub_torch.py transcript_en.json -o output/dubbed/

  # Test on first 5 segments
  python synthesize_dub_torch.py transcript_en.json --limit 5
        """
    )
    parser.add_argument("transcript", help="Path to translated transcript JSON file")
    parser.add_argument("--clips", "-c", help="Path to clips_metadata.json (auto-detected if not specified)")
    parser.add_argument("--output-dir", "-o", help="Output directory for audio files")
    parser.add_argument("--model", "-m", default="Qwen/Qwen3-TTS-12Hz-1.7B-Base",
                        help="TTS model (default: Qwen3-TTS 1.7B Base)")
    parser.add_argument("--limit", "-l", type=int, help="Only synthesize first N segments (for testing)")
    parser.add_argument("--gap", "-g", type=float, default=0.3,
                        help="Gap between segments in seconds (default: 0.3)")
    parser.add_argument("--no-trim", action="store_true", help="Don't trim silence from generated audio")
    parser.add_argument("--temperature", "-t", type=float, default=0.7,
                        help="Sampling temperature (default: 0.7, lower = more stable)")
    parser.add_argument("--device", "-d", default="cuda:0",
                        help="Device to run on (default: cuda:0)")
    parser.add_argument("--ref-text", action="store_true",
                        help="Pass reference transcript text to help voice cloning (ICL mode)")
    args = parser.parse_args()

    # Load transcript
    transcript_path = Path(args.transcript)
    if not transcript_path.exists():
        print(f"Error: {transcript_path} not found")
        return 1

    with open(transcript_path) as f:
        transcript = json.load(f)

    segments = transcript.get('segments', [])
    if not segments:
        print("Error: No segments found in transcript")
        return 1

    # Check for translation metadata
    translation_info = transcript.get('translation', {})
    target_lang = translation_info.get('target_language', 'en')

    # Limit segments if requested
    if args.limit:
        segments = segments[:args.limit]

    # Find clips metadata
    if args.clips:
        clips_path = Path(args.clips)
    else:
        # Auto-detect: look for clips_metadata.json in parent directories
        transcript_dir = transcript_path.parent
        possible_paths = [
            transcript_dir / "clips_metadata.json",
            transcript_dir.parent / f"{transcript_dir.stem}_speaker_clips" / "clips_metadata.json",
            transcript_dir.parent / "clips_metadata.json",
        ]
        # Also check for *_speaker_clips sibling folders
        for sibling in transcript_dir.parent.glob("*_speaker_clips"):
            possible_paths.append(sibling / "clips_metadata.json")

        clips_path = None
        for p in possible_paths:
            if p.exists():
                clips_path = p
                break

        if clips_path is None:
            print("Error: Could not find clips_metadata.json")
            print("Specify path with --clips or run extract_speaker_clips.py first")
            return 1

    print(f"Loading clips metadata: {clips_path}")
    speaker_refs = load_speaker_references(clips_path)
    print(f"Found {len(speaker_refs)} speakers with reference clips")

    # Verify all speakers in transcript have references
    transcript_speakers = set(seg.get('speaker', 0) for seg in segments)
    missing_speakers = transcript_speakers - set(speaker_refs.keys())
    if missing_speakers:
        print(f"Warning: No reference clips for speakers: {missing_speakers}")
        print("These segments will be skipped")

    # Set output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = transcript_path.parent / f"{transcript_path.stem}_dubbed"

    output_dir.mkdir(parents=True, exist_ok=True)
    segments_dir = output_dir / "segments"
    segments_dir.mkdir(exist_ok=True)

    # Map ISO language codes to Qwen3-TTS language names
    lang_map = {
        'en': 'English',
        'zh': 'Chinese',
        'ja': 'Japanese',
        'ko': 'Korean',
        'de': 'German',
        'fr': 'French',
        'ru': 'Russian',
        'pt': 'Portuguese',
        'es': 'Spanish',
        'it': 'Italian',
    }
    lang_name = lang_map.get(target_lang, 'Auto')

    # Resolve reference audio paths
    speaker_ref_paths = {}
    for speaker, ref in speaker_refs.items():
        ref_path = Path(ref['file'])
        if not ref_path.is_absolute():
            # Try relative to clips metadata or project root
            if (clips_path.parent / ref_path.name).exists():
                ref_path = clips_path.parent / ref_path.name
            elif (Path.cwd() / ref['file']).exists():
                ref_path = Path.cwd() / ref['file']

        if not ref_path.exists():
            print(f"Warning: Reference audio not found: {ref_path}")
            continue

        speaker_ref_paths[speaker] = {
            'path': str(ref_path),
            'text': ref['text'],
            'duration': ref['duration'],
        }
        print(f"  Speaker {speaker}: {ref_path.name} ({ref['duration']:.1f}s)")

    # Count active speakers
    active_speakers = set(
        seg.get('speaker', 0) for seg in segments
        if not (seg.get('text', '').startswith('[') and seg.get('text', '').endswith(']'))
    )
    active_speakers = active_speakers & set(speaker_ref_paths.keys())

    print(f"\nTarget language: {lang_name}")
    print(f"Segments: {len(segments)}")
    print(f"Active speakers: {len(active_speakers)}")
    print()

    # Load TTS model
    print(f"Loading TTS model: {args.model}")
    load_start = time.perf_counter()

    from qwen_tts import Qwen3TTSModel

    tts_model = Qwen3TTSModel.from_pretrained(
        args.model,
        device_map=args.device,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",  # Works on ROCm with AOTriton
    )
    sample_rate = 24000  # Qwen3-TTS always outputs 24kHz

    load_time = time.perf_counter() - load_start
    print(f"Model loaded in {load_time:.2f}s")

    # Pre-create voice clone prompts for each speaker (more efficient for batch)
    print("\nCreating voice clone prompts...")
    speaker_prompts = {}
    for speaker, ref_info in speaker_ref_paths.items():
        ref_text = ref_info['text'] if args.ref_text else None
        prompt = tts_model.create_voice_clone_prompt(
            ref_audio=ref_info['path'],
            ref_text=ref_text,
            x_vector_only_mode=not args.ref_text,  # Use ICL mode if ref_text provided
        )
        speaker_prompts[speaker] = prompt
        mode = "ICL" if args.ref_text else "x-vector"
        print(f"  Speaker {speaker}: {mode} mode")

    print()
    print(f"Synthesizing {len(segments)} segments...")
    print(f"Temperature: {args.temperature}")
    print("=" * 60)

    synth_start = time.perf_counter()
    generated_segments = []

    for i, seg in enumerate(segments):
        text = seg.get('text', '')
        speaker = seg.get('speaker', 0)

        # Skip non-speech markers
        if text.startswith('[') and text.endswith(']'):
            print(f"[{i+1}/{len(segments)}] Skipping: {text}")
            continue

        # Skip empty text
        if not text.strip():
            print(f"[{i+1}/{len(segments)}] Skipping empty segment")
            continue

        # Skip if no reference for this speaker
        if speaker not in speaker_prompts:
            print(f"[{i+1}/{len(segments)}] Skipping (no ref): Speaker {speaker}")
            continue

        print(f"[{i+1}/{len(segments)}] Speaker {speaker}: {text[:50]}...")

        seg_start = time.perf_counter()

        # Generate speech using pre-computed voice clone prompt
        wavs, sr = tts_model.generate_voice_clone(
            text=text,
            language=lang_name,
            voice_clone_prompt=speaker_prompts[speaker],
            temperature=args.temperature,
            do_sample=True,
            top_k=50,
            top_p=1.0,
            repetition_penalty=1.05,
        )

        # wavs is a list with one element per input text
        audio = wavs[0]

        # Trim silence
        if not args.no_trim:
            audio = trim_silence(audio, sample_rate)

        seg_time = time.perf_counter() - seg_start
        duration = len(audio) / sample_rate

        # Save individual segment
        seg_filename = f"segment_{i:04d}_speaker_{speaker}.wav"
        seg_path = segments_dir / seg_filename
        sf.write(str(seg_path), audio, sample_rate)

        generated_segments.append({
            'index': i,
            'speaker': speaker,
            'text': text,
            'original_text': seg.get('original_text', ''),
            'audio_file': str(seg_path),
            'duration': duration,
            'synth_time': seg_time,
        })

        print(f"         -> {duration:.2f}s audio in {seg_time:.2f}s")

    synth_time = time.perf_counter() - synth_start

    print()
    print("=" * 60)
    print(f"Synthesis completed in {synth_time:.2f}s")
    print(f"Generated {len(generated_segments)} segments")
    print()

    # Combine all segments with gaps
    print("Combining segments...")
    gap_samples = int(args.gap * sample_rate)
    gap_audio = np.zeros(gap_samples, dtype=np.float32)

    combined_parts = []
    for seg_info in generated_segments:
        audio, _ = sf.read(seg_info['audio_file'])
        combined_parts.append(audio)
        combined_parts.append(gap_audio)

    # Remove trailing gap
    if combined_parts:
        combined_parts = combined_parts[:-1]

    combined_audio = np.concatenate(combined_parts) if combined_parts else np.array([])
    combined_duration = len(combined_audio) / sample_rate

    # Save combined audio
    combined_path = output_dir / f"{transcript_path.stem}_dubbed.wav"
    sf.write(str(combined_path), combined_audio, sample_rate)
    print(f"Combined audio: {combined_path} ({combined_duration:.2f}s)")

    # Save metadata
    metadata = {
        'source_transcript': str(transcript_path.absolute()),
        'clips_metadata': str(clips_path.absolute()),
        'target_language': target_lang,
        'model': args.model,
        'sample_rate': sample_rate,
        'gap_seconds': args.gap,
        'synth_seconds': round(synth_time, 2),
        'combined_duration': round(combined_duration, 2),
        'segments': generated_segments,
    }

    metadata_path = output_dir / "dub_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(f"Metadata: {metadata_path}")
    print()
    print("=" * 60)
    print(f"Output directory: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()