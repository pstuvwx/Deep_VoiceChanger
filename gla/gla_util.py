import numpy as np

class GLA:
    def __init__(self, wave_len=254, wave_dif=64, buffer_size=5, loop_num=5, window=np.hanning(254)):
        self.wave_len = wave_len
        self.wave_dif = wave_dif
        self.buffer_size = buffer_size
        self.loop_num = loop_num
        self.window = window

        self.wave_buf = np.zeros(wave_len+wave_dif, dtype=float)
        self.overwrap_buf = np.zeros(wave_dif*buffer_size+(wave_len-wave_dif), dtype=float)
        self.spectrum_buffer = np.ones((self.buffer_size, self.wave_len), dtype=complex)
        self.absolute_buffer = np.ones((self.buffer_size, self.wave_len), dtype=complex)
        
        self.phase = np.zeros(self.wave_len, dtype=complex)
        self.phase += np.random.random(self.wave_len)-0.5 + np.random.random(self.wave_len)*1j - 0.5j
        self.phase[self.phase == 0] = 1
        self.phase /= np.abs(self.phase)

    def inverse(self, spectrum, in_phase=None):
        spectrum = spectrum.astype(complex)
        if in_phase is None:
            in_phase = self.phase
        self.spectrum_buffer[-1] = spectrum * in_phase
        self.absolute_buffer[-1] = spectrum

        for _ in range(self.loop_num):
            self.overwrap_buf *= 0
            waves = np.fft.ifft(self.spectrum_buffer, axis=1).real
            last = self.spectrum_buffer

            for i in range(self.buffer_size):
                self.overwrap_buf[i*self.wave_dif:i*self.wave_dif+self.wave_len] += waves[i]
            waves = np.vstack([self.overwrap_buf[i*self.wave_dif:i*self.wave_dif+self.wave_len]*self.window for i in range(self.buffer_size)])

            spectrum = np.fft.fft(waves, axis=1)
            self.spectrum_buffer = self.absolute_buffer * spectrum / (np.abs(spectrum)+1e-10)
            self.spectrum_buffer += 0.5 * (self.spectrum_buffer - last)

        waves = np.fft.ifft(self.spectrum_buffer[0]).real
        self.absolute_buffer = np.roll(self.absolute_buffer, -1, axis=0)
        self.spectrum_buffer = np.roll(self.spectrum_buffer, -1, axis=0)

        self.wave_buf = np.roll(self.wave_buf, -self.wave_dif)
        self.wave_buf[-self.wave_dif:] = 0
        self.wave_buf[self.wave_dif:] += waves
        return self.wave_buf[:self.wave_dif]*0.5

if __name__ == "__main__":
    import tqdm
    import scipy.io.wavfile as wav
    def load(path):
        bps, data = wav.read(path)
        if len(data.shape) != 1:
            data = data[:,0] + data[:,1]
        return bps, data

    def save(path, bps, data):
        if data.dtype != np.int16:
            data = data.astype(np.int16)
        data = np.reshape(data, -1)
        wav.write(path, bps, data)


    bps, wave = load(input("path..."))

    wave_len = 254
    wave_dif = 64
    window = np.hanning(wave_len)
    num = len(wave)//wave_dif-3
    spl = np.vstack([np.fft.fft(wave[i:i+wave_len]*window) for i in range(0, wave_dif*num, wave_dif)])
    absolute = np.abs(spl)
    spl[absolute < 1] = 1

    gla = GLA()

    dst = [gla.inverse(a) for a in tqdm.tqdm(absolute)]

    w = np.hstack(dst)

    save("w.wav", bps, w)
