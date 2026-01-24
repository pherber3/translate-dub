#!/usr/bin/env python3
"""Translate a transcript JSON to another language.

Takes a transcript JSON (from transcribe.py) and translates all text segments
to the target language, preserving all timing and speaker information.

Supports parallel translation by speaker using multiprocessing.
"""

import argparse
import json
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path


def translate_text(model, tokenizer, text: str, source_lang: str, target_lang: str) -> str:
    """Translate a single text segment."""
    from mlx_lm import generate

    prompt = f"Translate this {source_lang} sentence to {target_lang}. Output only the translation, nothing else.\n\n{text}"

    result = generate(
        model,
        tokenizer,
        prompt=prompt,
        max_tokens=200,
    )

    # Clean output
    result = result.strip()
    if "<end_of_turn>" in result:
        result = result.split("<end_of_turn>")[0].strip()
    if "\n" in result:
        result = result.split("\n")[0].strip()
    if result.startswith('"') and result.endswith('"'):
        result = result[1:-1]

    return result.strip()


def translate_speaker_segments(
    speaker: int,
    segments_with_indices: list,
    source_lang: str,
    target_lang: str,
    model_name: str,
) -> list:
    """Translate all segments for a single speaker. Runs in separate process."""
    from mlx_lm import load

    # Load model in this process
    model, tokenizer = load(model_name)

    results = []
    for idx, seg in segments_with_indices:
        text = seg.get('text', '')

        # Skip non-speech markers
        if text.startswith('[') and text.endswith(']'):
            translated_text = text
        else:
            translated_text = translate_text(model, tokenizer, text, source_lang, target_lang)

        translated_seg = seg.copy()
        translated_seg['text'] = translated_text
        translated_seg['original_text'] = text
        results.append((idx, translated_seg))

        print(f"[Speaker {speaker}] {text[:40]}... -> {translated_text[:40]}...")

    return results


# Language code to full name mapping
LANG_NAMES = {
    "en": "English", "de": "German", "fr": "French", "es": "Spanish",
    "it": "Italian", "pt": "Portuguese", "ru": "Russian", "zh": "Chinese",
    "ja": "Japanese", "ko": "Korean", "nl": "Dutch", "pl": "Polish",
}


def main():
    parser = argparse.ArgumentParser(
        description="Translate a transcript JSON to another language",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Translate French transcript to English
  python translate_transcript.py transcript.json --source fr --target en

  # Translate to German with custom output path
  python translate_transcript.py transcript.json -s en -t de -o transcript_de.json
        """
    )
    parser.add_argument("transcript", help="Path to transcript JSON file")
    parser.add_argument("--source", "-s", required=True, help="Source language code (e.g., fr, en, de)")
    parser.add_argument("--target", "-t", required=True, help="Target language code (e.g., en, de, fr)")
    parser.add_argument("--output", "-o", help="Output JSON file path (default: transcript_<target>.json)")
    parser.add_argument("--model", "-m", default="mlx-community/translategemma-12b-it-4bit",
                        help="Translation model (default: translategemma-12b-it-4bit)")
    parser.add_argument("--limit", "-l", type=int, help="Only translate first N segments (for testing)")
    parser.add_argument("--parallel", "-p", action="store_true", help="Translate speakers in parallel (uses more memory)")
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

    # Limit segments if requested
    if args.limit:
        segments = segments[:args.limit]

    # Set output path
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = transcript_path.with_stem(f"{transcript_path.stem}_{args.target}")

    # Get full language names
    source_name = LANG_NAMES.get(args.source, args.source)
    target_name = LANG_NAMES.get(args.target, args.target)

    # Count speakers
    speakers = set(seg.get('speaker', 0) for seg in segments)

    print(f"Translating: {source_name} -> {target_name}")
    print(f"Segments: {len(segments)}")
    print(f"Speakers: {len(speakers)}")
    if args.parallel:
        print(f"Mode: Parallel (one model per speaker)")
    print()

    translate_start = time.perf_counter()

    if args.parallel and len(speakers) > 1:
        # Group segments by speaker, keeping original indices
        by_speaker = defaultdict(list)
        for idx, seg in enumerate(segments):
            speaker = seg.get('speaker', 0)
            by_speaker[speaker].append((idx, seg))

        print(f"Spawning {len(speakers)} parallel processes...")
        print()

        # Run translation in parallel per speaker
        all_results = []
        with ProcessPoolExecutor(max_workers=len(speakers)) as executor:
            futures = {
                executor.submit(
                    translate_speaker_segments,
                    speaker,
                    segs,
                    source_name,
                    target_name,
                    args.model,
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
        translated_segments = [seg for _, seg in all_results]

    else:
        # Sequential mode
        from mlx_lm import load

        print(f"Loading model: {args.model}")
        load_start = time.perf_counter()
        model, tokenizer = load(args.model)
        load_time = time.perf_counter() - load_start
        print(f"Model loaded in {load_time:.2f}s")
        print()

        translated_segments = []
        for i, seg in enumerate(segments):
            text = seg.get('text', '')

            # Skip non-speech markers like [Music], [Silence]
            if text.startswith('[') and text.endswith(']'):
                translated_text = text  # Keep as-is
            else:
                translated_text = translate_text(model, tokenizer, text, source_name, target_name)

            # Copy segment with translated text
            translated_seg = seg.copy()
            translated_seg['text'] = translated_text
            translated_seg['original_text'] = text
            translated_segments.append(translated_seg)

            # Progress
            print(f"[{i+1}/{len(segments)}] {text[:50]}...")
            print(f"        -> {translated_text[:50]}...")

    translate_time = time.perf_counter() - translate_start

    # Build output
    output = transcript.copy()
    output['segments'] = translated_segments
    output['translation'] = {
        'source_language': args.source,
        'target_language': args.target,
        'model': args.model,
        'translate_seconds': round(translate_time, 2),
    }

    # Save
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print()
    print("=" * 60)
    print(f"Translation completed in {translate_time:.2f}s")
    print(f"Saved to: {output_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
