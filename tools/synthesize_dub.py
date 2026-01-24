#!/usr/bin/env python3
"""Synthesize dubbed audio from translated transcript using voice cloning.

Takes a translated transcript JSON and speaker clips metadata, then generates
speech for each segment using Qwen3-TTS voice cloning with the appropriate
speaker's voice.

Output: individual segment WAVs + combined audio file.

Supports parallel synthesis by speaker using multiprocessing.
"""

import argparse
import json
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf


def trim_silence(audio: np.ndarray, sr: int, top_db: int = 25) -> np.ndarray:
    """Trim leading and trailing silence from audio."""
    trimmed, _ = librosa.effects.trim(audio, top_db=top_db)
    return trimmed


def synthesize_speaker_segments(
    speaker: int,
    segments_with_indices: list,
    ref_audio_path: str,
    lang_code: str,
    model_name: str,
    output_dir: str,
    trim: bool = True,
) -> list:
    """Synthesize all segments for a single speaker. Runs in separate process."""
    import mlx.core as mx
    from mlx_audio.tts import load as load_tts

    # Load model in this process
    tts_model = load_tts(model_name)
    sample_rate = tts_model.sample_rate

    # Load reference audio once for this speaker
    ref_audio_np, _ = librosa.load(ref_audio_path, sr=sample_rate)
    ref_audio = mx.array(ref_audio_np)

    output_path = Path(output_dir)
    results = []

    for idx, seg in segments_with_indices:
        text = seg.get('text', '')

        # Skip non-speech markers
        if text.startswith('[') and text.endswith(']'):
            continue

        seg_start = time.perf_counter()

        # Generate speech
        gen_results = list(tts_model.generate(
            text=text,
            lang_code=lang_code,
            ref_audio=ref_audio,
        ))

        # Concatenate results if text was split
        if len(gen_results) == 1:
            audio = np.array(gen_results[0].audio)
        else:
            audio = np.concatenate([np.array(r.audio) for r in gen_results])

        # Trim silence
        if trim:
            audio = trim_silence(audio, sample_rate)

        seg_time = time.perf_counter() - seg_start
        duration = len(audio) / sample_rate

        # Save individual segment
        seg_filename = f"segment_{idx:04d}_speaker_{speaker}.wav"
        seg_path = output_path / seg_filename
        sf.write(str(seg_path), audio, sample_rate)

        results.append((idx, {
            'index': idx,
            'speaker': speaker,
            'text': text,
            'original_text': seg.get('original_text', ''),
            'audio_file': str(seg_path),
            'duration': duration,
            'synth_time': seg_time,
        }))

        print(f"[Speaker {speaker}] {text[:40]}... -> {duration:.2f}s")

    return results


def load_speaker_references(clips_metadata_path: Path) -> dict:
    """Load ALL speaker reference clips from metadata, sorted by rank.

    Returns dict mapping speaker_id -> list of {file, text, duration, rank}
    Clips are sorted by rank (best first).
    """
    with open(clips_metadata_path) as f:
        metadata = json.load(f)

    references = defaultdict(list)
    for clip in metadata['clips']:
        speaker = clip['speaker']
        references[speaker].append({
            'file': clip['file'],
            'text': clip['text'],
            'duration': clip['duration'],
            'rank': clip['rank'],
        })

    # Sort each speaker's clips by rank (lower = better)
    for speaker in references:
        references[speaker].sort(key=lambda x: x['rank'])

    return dict(references)


def estimate_max_tokens(text: str, multiplier: float = 3.0) -> int:
    """Estimate reasonable max_tokens based on text length.

    At 12Hz TTS output:
    - Normal speech is ~150 words/min = 2.5 words/sec
    - Average word is ~5 chars, so ~12.5 chars/sec
    - At 12 tokens/sec output, that's roughly 1 token per char

    We use a generous multiplier to avoid cutting off speech.
    The max_tokens is a timeout, not a target - better to be too high
    than to truncate the output.
    """
    base_tokens = len(text)
    # Minimum of 100 tokens, apply multiplier
    return max(100, int(base_tokens * multiplier))


def generate_with_fallback(
    tts_model,
    text: str,
    lang_code: str,
    ref_audios: list,  # List of (clip_info, mx.array) tuples, sorted by rank
    max_tokens: int | None = None,
    temperature: float = 0.9,
    use_ref_text: bool = False,
) -> tuple:
    """Generate TTS with fallback to alternate reference clips on EOS failure.

    Args:
        tts_model: Loaded TTS model
        text: Text to synthesize
        lang_code: Language code for TTS
        ref_audios: List of (clip_info dict, ref_audio mx.array) tuples
        max_tokens: Max tokens before considering it a failure (auto-estimated if None)
        temperature: Sampling temperature (lower = more deterministic)
        use_ref_text: Whether to pass reference text to help with voice cloning

    Returns:
        (audio_array, token_count, clip_used, hit_eos)
    """
    import numpy as np

    if max_tokens is None:
        max_tokens = estimate_max_tokens(text)

    best_result = None
    best_clip = None

    for clip_info, ref_audio in ref_audios:
        # Optionally pass reference text to help model align
        ref_text = clip_info.get('text') if use_ref_text else None

        results = list(tts_model.generate(
            text=text,
            lang_code=lang_code,
            ref_audio=ref_audio,
            ref_text=ref_text,
            max_tokens=max_tokens,
            temperature=temperature,
            verbose=False,
        ))

        if not results:
            continue

        # Concatenate if text was split
        if len(results) == 1:
            audio = np.array(results[0].audio)
            token_count = results[0].token_count
        else:
            audio = np.concatenate([np.array(r.audio) for r in results])
            token_count = sum(r.token_count for r in results)

        hit_eos = token_count < max_tokens

        if hit_eos:
            # Success! Return immediately
            return audio, token_count, clip_info, True

        # Keep track of best attempt in case all fail
        if best_result is None:
            best_result = audio
            best_clip = clip_info

    # All clips failed to hit EOS - return best attempt (rank-1)
    if best_result is not None:
        return best_result, max_tokens, best_clip, False

    # No results at all
    return None, 0, None, False


def main():
    parser = argparse.ArgumentParser(
        description="Synthesize dubbed audio from translated transcript",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with auto-detected clips
  python synthesize_dub.py transcript_en.json

  # Specify clips metadata explicitly
  python synthesize_dub.py transcript_en.json --clips clips_metadata.json

  # Custom output directory
  python synthesize_dub.py transcript_en.json -o output/dubbed/

  # Test on first 5 segments
  python synthesize_dub.py transcript_en.json --limit 5
        """
    )
    parser.add_argument("transcript", help="Path to translated transcript JSON file")
    parser.add_argument("--clips", "-c", help="Path to clips_metadata.json (auto-detected if not specified)")
    parser.add_argument("--output-dir", "-o", help="Output directory for audio files")
    parser.add_argument("--model", "-m", default="mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16",
                        help="TTS model (default: Qwen3-TTS 1.7B Base)")
    parser.add_argument("--limit", "-l", type=int, help="Only synthesize first N segments (for testing)")
    parser.add_argument("--gap", "-g", type=float, default=0.3,
                        help="Gap between segments in seconds (default: 0.3)")
    parser.add_argument("--no-trim", action="store_true", help="Don't trim silence from generated audio")
    parser.add_argument("--parallel", "-p", action="store_true",
                        help="Synthesize speakers in parallel (uses more memory)")
    parser.add_argument("--max-tokens", type=int,
                        help="Max tokens per segment (auto-estimated if not specified)")
    parser.add_argument("--temperature", "-t", type=float, default=0.7,
                        help="Sampling temperature (default: 0.7, lower = more stable)")
    parser.add_argument("--ref-text", action="store_true",
                        help="Pass reference transcript text to help voice cloning")
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
        # or sibling *_speaker_clips directories
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

    # Language code for TTS (lowercase full name)
    from pipeline_mlx.language_utils import iso_to_full
    lang_name = iso_to_full(target_lang).lower()

    # Resolve reference audio paths for ALL clips per speaker
    speaker_clip_paths = defaultdict(list)  # speaker -> list of (clip_info, resolved_path)
    for speaker, clips in speaker_refs.items():
        for clip in clips:
            ref_path = Path(clip['file'])
            if not ref_path.is_absolute():
                # Try relative to clips metadata or project root
                if (clips_path.parent / ref_path.name).exists():
                    ref_path = clips_path.parent / ref_path.name
                elif (Path.cwd() / clip['file']).exists():
                    ref_path = Path.cwd() / clip['file']

            if not ref_path.exists():
                print(f"Warning: Reference audio not found: {ref_path}")
                continue

            speaker_clip_paths[speaker].append((clip, str(ref_path)))

        if speaker_clip_paths[speaker]:
            primary = speaker_clip_paths[speaker][0]
            print(f"  Speaker {speaker}: {len(speaker_clip_paths[speaker])} clips, primary: {Path(primary[1]).name} ({primary[0]['duration']:.1f}s)")

    # Count speakers in segments
    active_speakers = set(seg.get('speaker', 0) for seg in segments
                          if not (seg.get('text', '').startswith('[') and seg.get('text', '').endswith(']')))
    active_speakers = active_speakers & set(speaker_clip_paths.keys())

    print(f"\nTarget language: {lang_name}")
    print(f"Segments: {len(segments)}")
    print(f"Active speakers: {len(active_speakers)}")
    if args.parallel:
        print(f"Mode: Parallel (one model per speaker)")
    print()

    synth_start = time.perf_counter()

    if args.parallel and len(active_speakers) > 1:
        # Group segments by speaker, keeping original indices
        by_speaker = defaultdict(list)
        for idx, seg in enumerate(segments):
            text = seg.get('text', '')
            # Skip non-speech markers
            if text.startswith('[') and text.endswith(']'):
                continue
            speaker = seg.get('speaker', 0)
            if speaker in speaker_ref_paths:
                by_speaker[speaker].append((idx, seg))

        print(f"Spawning {len(active_speakers)} parallel processes...")
        print("=" * 60)

        # Run synthesis in parallel per speaker
        all_results = []
        with ProcessPoolExecutor(max_workers=len(active_speakers)) as executor:
            futures = {
                executor.submit(
                    synthesize_speaker_segments,
                    speaker,
                    segs,
                    speaker_ref_paths[speaker],
                    lang_name,
                    args.model,
                    str(segments_dir),
                    not args.no_trim,
                ): speaker
                for speaker, segs in by_speaker.items()
            }

            for future in as_completed(futures):
                speaker = futures[future]
                results = future.result()
                all_results.extend(results)
                print(f"[Speaker {speaker}] Done ({len(results)} segments)")

        # Sort by original index to restore order
        all_results.sort(key=lambda x: x[0])
        generated_segments = [seg_info for _, seg_info in all_results]

    else:
        # Sequential mode
        print(f"Loading TTS model: {args.model}")
        load_start = time.perf_counter()

        import mlx.core as mx
        from mlx_audio.tts import load as load_tts

        tts_model = load_tts(args.model)
        sample_rate = tts_model.sample_rate  # 24000
        load_time = time.perf_counter() - load_start
        print(f"Model loaded in {load_time:.2f}s")

        # Pre-load ALL reference audio clips as MLX arrays
        speaker_ref_audios = {}  # speaker -> list of (clip_info, mx.array)
        for speaker, clip_paths in speaker_clip_paths.items():
            speaker_ref_audios[speaker] = []
            for clip_info, ref_path in clip_paths:
                ref_audio_np, _ = librosa.load(ref_path, sr=sample_rate)
                speaker_ref_audios[speaker].append((clip_info, mx.array(ref_audio_np)))

        print()
        print(f"Synthesizing {len(segments)} segments...")
        print(f"Temperature: {args.temperature}")
        if args.ref_text:
            print("Using reference text: yes")
        if args.max_tokens:
            print(f"Max tokens: {args.max_tokens}")
        else:
            print("Max tokens: auto-estimated per segment")
        print("=" * 60)

        generated_segments = []
        fallback_count = 0
        eos_failure_count = 0

        for i, seg in enumerate(segments):
            text = seg.get('text', '')
            speaker = seg.get('speaker', 0)

            # Skip non-speech markers
            if text.startswith('[') and text.endswith(']'):
                print(f"[{i+1}/{len(segments)}] Skipping: {text}")
                continue

            # Skip if no reference for this speaker
            if speaker not in speaker_ref_audios:
                print(f"[{i+1}/{len(segments)}] Skipping (no ref): Speaker {speaker}")
                continue

            print(f"[{i+1}/{len(segments)}] Speaker {speaker}: {text[:50]}...")

            # Generate speech with fallback
            seg_start = time.perf_counter()

            ref_audios = speaker_ref_audios[speaker]
            audio, token_count, clip_used, hit_eos = generate_with_fallback(
                tts_model,
                text,
                lang_name,
                ref_audios,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                use_ref_text=args.ref_text,
            )

            if audio is None:
                print(f"         -> FAILED (no audio generated)")
                continue

            # Trim silence
            if not args.no_trim:
                audio = trim_silence(audio, sample_rate)

            seg_time = time.perf_counter() - seg_start
            duration = len(audio) / sample_rate

            # Track which clip was used
            clip_rank = clip_used['rank'] if clip_used else 0
            if clip_rank > 1:
                fallback_count += 1
            if not hit_eos:
                eos_failure_count += 1

            # Status indicator
            status = ""
            if clip_rank > 1:
                status += f" [fallback: rank {clip_rank}]"
            if not hit_eos:
                status += " [no EOS]"

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
                'ref_clip_rank': clip_rank,
                'hit_eos': hit_eos,
                'token_count': token_count,
            })

            print(f"         -> {duration:.2f}s audio ({token_count} tokens) in {seg_time:.2f}s{status}")

        # Summary of fallbacks
        if fallback_count > 0 or eos_failure_count > 0:
            print()
            print(f"Fallback summary: {fallback_count} used alternate clips, {eos_failure_count} hit max_tokens")

    synth_time = time.perf_counter() - synth_start

    print()
    print("=" * 60)
    print(f"Synthesis completed in {synth_time:.2f}s")
    print(f"Generated {len(generated_segments)} segments")
    print()

    # Combine all segments with gaps
    # Qwen3-TTS always outputs at 24kHz
    sample_rate = 24000
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
