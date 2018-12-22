import numpy as np
import cupy as cp
import tqdm

class GLA_GPU:
    def __init__(self, parallel, wave_len=254, wave_dif=64, buffer_size=5, loop_num=5, window=np.hanning(254)):
        self.wave_len = wave_len
        self.wave_dif = wave_dif
        self.buffer_size = buffer_size
        self.loop_num = loop_num
        self.parallel = parallel
        self.window = cp.array([window for _ in range(parallel)])

        self.wave_buf = cp.zeros((parallel, wave_len+wave_dif), dtype=float)
        self.overwrap_buf = cp.zeros((parallel, wave_dif*buffer_size+(wave_len-wave_dif)), dtype=float)
        self.spectrum_buffer = cp.ones((parallel, self.buffer_size, self.wave_len), dtype=complex)
        self.absolute_buffer = cp.ones((parallel, self.buffer_size, self.wave_len), dtype=complex)
        
        self.phase = cp.zeros((parallel, self.wave_len), dtype=complex)
        self.phase += cp.random.random((parallel, self.wave_len))-0.5 + cp.random.random((parallel, self.wave_len))*1j - 0.5j
        self.phase[self.phase == 0] = 1
        self.phase /= cp.abs(self.phase)

    def inverse(self, spectrum, in_phase=None):
        if in_phase is None:
            in_phase = self.phase
        else:
            in_phase = cp.array(in_phase)
        spectrum = cp.array(spectrum)
        self.spectrum_buffer[:, -1] = spectrum * in_phase
        self.absolute_buffer[:, -1] = spectrum

        for _ in range(self.loop_num):
            self.overwrap_buf *= 0
            waves = cp.fft.ifft(self.spectrum_buffer, axis=2).real
            last = self.spectrum_buffer

            for i in range(self.buffer_size):
                self.overwrap_buf[:,i*self.wave_dif:i*self.wave_dif+self.wave_len] += waves[:,i]
            waves = cp.stack([self.overwrap_buf[:, i*self.wave_dif:i*self.wave_dif+self.wave_len]*self.window for i in range(self.buffer_size)], axis=1)

            spectrum = cp.fft.fft(waves, axis=2)
            self.spectrum_buffer = self.absolute_buffer * spectrum / (cp.abs(spectrum)+1e-10)
            self.spectrum_buffer += 0.5 * (self.spectrum_buffer - last)

        dst = cp.asnumpy(self.spectrum_buffer[:, 0])
        self.absolute_buffer = cp.roll(self.absolute_buffer, -1, axis=1)
        self.spectrum_buffer = cp.roll(self.spectrum_buffer, -1, axis=1)

        return dst
    
    def auto_inverse(self, whole_spectrum):
        whole_spectrum = np.copy(whole_spectrum).astype(complex)
        whole_spectrum[whole_spectrum < 1] = 1
        overwrap = self.buffer_size * 2
        height = whole_spectrum.shape[0]
        parallel_dif = (height-overwrap) // self.parallel
        if height < self.parallel*overwrap:
            raise Exception('voice length is too small to use gpu, or parallel number is too big')

        spec = [self.inverse(whole_spectrum[range(i, i+parallel_dif*self.parallel, parallel_dif), :]) for i in tqdm.tqdm(range(parallel_dif+overwrap))]
        spec = spec[overwrap:]
        spec = np.concatenate(spec, axis=1)
        spec = spec.reshape(-1, self.wave_len)

        #Below code don't consider wave_len and wave_dif, I'll fix.
        wave = np.fft.ifft(spec, axis=1).real
        pad = np.zeros((wave.shape[0], 2), dtype=float)
        wave = np.concatenate([wave, pad], axis=1)

        dst = np.zeros((wave.shape[0]+3)*self.wave_dif, dtype=float)
        for i in range(4):
            w = wave[range(i, wave.shape[0], 4),:]
            w = w.reshape(-1)
            dst[i*self.wave_dif:i*self.wave_dif+len(w)] += w
        return dst*0.5


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

    path = input("enter wave path...")
    bps, wave = load(path)
    cp.cuda.Device(input('gpu number...')).use()


    wave_len = 254
    wave_dif = 64
    window = np.hanning(wave_len)
    spec = np.vstack([np.fft.fft(wave[i:i+wave_len]*window) for i in range(0, len(wave) - 3 * wave_dif, wave_dif)]).reshape(-1, wave_len)
    spec = np.abs(spec)

    gla = GLA_GPU(128)

    w = gla.auto_inverse(spec)

    save(path+'gla.wav', bps, w)
