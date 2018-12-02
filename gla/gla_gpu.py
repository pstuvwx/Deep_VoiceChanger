import numpy as np
import cupy as cp

class GLA_GPU:
    def __init__(self, parallel, wave_len=254, wave_dif=64, buffer_size=5, loop_num=5, window=np.hanning(254)):
        self.wave_len = wave_len
        self.wave_dif = wave_dif
        self.buffer_size = buffer_size
        self.loop_num = loop_num
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
            self.spectrum_buffer = self.absolute_buffer * spectrum / cp.abs(spectrum)
            self.spectrum_buffer += 0.5 * (self.spectrum_buffer - last)

        waves = cp.fft.ifft(self.spectrum_buffer[:, 0]).real
        self.absolute_buffer = cp.roll(self.absolute_buffer, -1, axis=1)
        self.spectrum_buffer = cp.roll(self.spectrum_buffer, -1, axis=1)

        self.wave_buf = cp.roll(self.wave_buf, -self.wave_dif, axis=1)
        self.wave_buf[:, -self.wave_dif:] = 0
        self.wave_buf[:, self.wave_dif:] += waves
        return cp.asnumpy(self.wave_buf[:, :self.wave_dif])

if __name__ == "__main__":
    import dataset
    import time

    bps, wave = dataset.load(input("wave path..."))
    cp.cuda.Device(input('gpu number...')).use()


    wave_len = 254
    wave_dif = 64
    parts_num = 1000
    window = np.hanning(wave_len)
    num = (len(wave)//wave_dif-3)//parts_num*parts_num
    spl = np.vstack([np.fft.fft(wave[i:i+wave_len]*window) for i in range(0, wave_dif*num, wave_dif)])
    spl[spl == 0] = 1
    absolute = np.abs(spl).astype(complex).reshape((-1, parts_num, wave_len))

    print(absolute.shape[0])
    gla = GLA_GPU(absolute.shape[0])

    start = time.time()
    dst = [gla.inverse(absolute[:, i, :]) for i in range(parts_num)]
    end = time.time()

    print('convert time per wave_dif', (end - start) / num, 'wave_dif time', wave_dif / bps)

    w = np.stack(dst, axis=1).reshape(-1)

    dataset.save("w.wav", bps, w)
