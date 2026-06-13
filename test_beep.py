import numpy as np
import scipy.io.wavfile as wav

def generate_and_save_chime():
    sample_rate = 44100
    
    def generate_wave(frequency, duration):
        t = np.linspace(0, duration, int(sample_rate * duration), False)
        wave = np.sin(2 * np.pi * frequency * t)
        
        # Smooth the edges to prevent popping
        fade_size = int(sample_rate * 0.015) 
        fade_in = np.linspace(0, 1, fade_size)
        fade_out = np.linspace(1, 0, fade_size)
        
        wave[:fade_size] *= fade_in
        wave[-fade_size:] *= fade_out
        
        return wave

    # Generate the pieces
    tone1 = generate_wave(440, 0.12)  # Low tone (120ms)
    silence = np.zeros(int(sample_rate * 0.06))  # 60ms pure silence gap
    tone2 = generate_wave(600, 0.15)  # High tone (150ms)
    
    # Combine into a single continuous array
    full_chime = np.concatenate([tone1, silence, tone2])
    
    # Convert to 16-bit PCM scale (-32768 to 32767)
    audio_data = (full_chime * 32767).astype(np.int16)
    
    # Duplicate for stereo (Left/Right channels) so it is robust for all players
    stereo_data = np.repeat(audio_data[:, np.newaxis], 2, axis=1)
    
    # Save the file
    wav.write('listening_chime.wav', sample_rate, stereo_data)

generate_and_save_chime()