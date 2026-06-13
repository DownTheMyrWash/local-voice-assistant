"""Helper script to test all English Edge-TTS voices and local Kokoro voices."""
import os
import asyncio
import edge_tts
from pc_server.tts import TTS

# The sample text each voice will say
TEST_TEXT = "Hello! This is a test sample of my new voice. How do I sound to you?"

# Core high-quality Kokoro voices included in voices.bin
KOKORO_VOICES = [
    "af_heart", "af_alloy", "af_aoede", "af_bella", "af_jessica", 
    "af_kore", "af_nicole", "af_nova", "af_river", "af_sarah", "af_sky",
    "am_adam", "am_echo", "am_eric", "am_fenrir", "am_liam", "am_michael", 
    "am_onyx", "am_puck", "am_santa",
    "bf_alice", "bf_emma", "bf_isabella", "bf_lily",
    "bm_daniel", "bm_fable", "bm_george", "bm_lewis"
]

async def get_english_edge_voices():
    """Dynamically pull all English local variants from Edge-TTS."""
    print("Fetching available English voices from Edge-TTS...")
    all_voices = await edge_tts.list_voices()
    
    english_voices = []
    for v in all_voices:
        if v["Locale"].startswith("en-"):
            english_voices.append((v["Name"], v["ShortName"]))
    return english_voices

def save_audio(filename: str, data: bytes):
    """Save the generated PCM WAV data to disk."""
    if not data or len(data) < 44:
        print(f"       ⚠️ Failed to generate audio for {filename}")
        return
    with open(filename, "wb") as f:
        f.write(data)

async def main():
    output_dir = "voice_samples"
    os.makedirs(output_dir, exist_ok=True)
    
    tts = TTS()
    # Grab the running async loop
    loop = asyncio.get_running_loop()
    
    print("\n" + "="*50)
    print("      STARTING TTS VOICE GENERATION SCRIPT")
    print("="*50)

    # ----------------------------------------------------------------
    # PHASE 1: TEST KOKORO OFFLINE VOICES
    # ----------------------------------------------------------------
    print(f"\n[1/2] Generating {len(KOKORO_VOICES)} Offline Kokoro Voices...")
    tts.mode = "offline"
    
    for voice in KOKORO_VOICES:
        print(f" -> Rendering Kokoro: {voice}")
        tts.offline_voice = voice
        
        # FIX: Push the blocking call to a background thread to prevent deadlock
        wav_bytes = await loop.run_in_executor(None, tts._generate_wav, TEST_TEXT)
        
        filepath = os.path.join(output_dir, f"offline_kokoro_{voice}.wav")
        save_audio(filepath, wav_bytes)

    # # ----------------------------------------------------------------
    # # PHASE 2: TEST EDGE ONLINE VOICES
    # # ----------------------------------------------------------------
    # edge_voices = await get_english_edge_voices()
    # print(f"\n[2/2] Generating {len(edge_voices)} Online English Edge Voices...")
    # tts.mode = "online"
    
    # for i, (voice_id, short_name) in enumerate(edge_voices, 1):
    #     clean_filename = short_name.replace("-", "_").strip()
    #     print(f" -> Rendering Edge ({i}/{len(edge_voices)}): {short_name}")
        
    #     tts.online_voice = voice_id
        
    #     # FIX: Push the blocking call to a background thread to prevent deadlock
    #     wav_bytes = await loop.run_in_executor(None, tts._generate_wav, TEST_TEXT)
        
    #     filepath = os.path.join(output_dir, f"online_edge_{clean_filename}.wav")
    #     save_audio(filepath, wav_bytes)

    print("\n" + "="*50)
    print(f" DONE! All audio samples saved to the folder: './{output_dir}'")
    print("="*50)

if __name__ == "__main__":
    asyncio.run(main())