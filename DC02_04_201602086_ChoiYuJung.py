from __future__ import print_function

import sys
import wave

from io import StringIO

import alsaaudio
import colorama
import numpy as np
import pyaudio

from reedsolo import RSCodec, ReedSolomonError
from termcolor import cprint
from pyfiglet import figlet_format

HANDSHAKE_START_HZ = 4100
HANDSHAKE_END_HZ = 6160

START_HZ = 1024
STEP_HZ = 256
BITS = 4

FEC_BYTES = 4


def stereo_to_mono(input_file, output_file):
    inp = wave.open(input_file, 'r')
    params = list(inp.getparams())
    params[0] = 1 # nchannels
    params[3] = 0 # nframes

    out = wave.open(output_file, 'w')
    out.setparams(tuple(params))

    frame_rate = inp.getframerate()
    frames = inp.readframes(inp.getnframes())
    data = np.fromstring(frames, dtype=np.int16)
    left = data[0::2]
    out.writeframes(left.tostring())

    inp.close()
    out.close()

def yield_chunks(input_file, interval):
    wav = wave.open(input_file)
    frame_rate = wav.getframerate()

    chunk_size = int(round(frame_rate * interval))
    total_size = wav.getnframes()

    while True:
        chunk = wav.readframes(chunk_size)
        if len(chunk) == 0:
            return

        yield frame_rate, np.fromstring(chunk, dtype=np.int16)

def dominant(frame_rate, chunk):
    #print("chunk",chunk)
    w = np.fft.fft(chunk)
    #print("w:",w)
    freqs = np.fft.fftfreq(len(chunk))
    #print("freqs:",freqs)
    peak_coeff = np.argmax(np.abs(w))
    #print("peak_coeff:",peak_coeff)
    peak_freq = freqs[peak_coeff]
    #print("peak_freq",peak_freq)
    #print(abs(peak_freq * frame_rate))
    return abs(peak_freq * frame_rate) # in Hz

def match(freq1, freq2):
    return abs(freq1 - freq2) < 20

def decode_bitchunks(chunk_bits, chunks):
    print(chunks)
    out_bytes = []

    next_read_chunk = 0
    next_read_bit = 0

    byte = 0
    bits_left = 8
    while next_read_chunk < len(chunks):
        can_fill = chunk_bits - next_read_bit
        #print("can:",can_fill)
        to_fill = min(bits_left, can_fill)
        #print("to:",to_fill)
        offset = chunk_bits - next_read_bit - to_fill
        #print("offset:",offset)
        byte <<= to_fill
        #print("byte:",byte)
        shifted = chunks[next_read_chunk] & (((1 << to_fill) - 1) << offset)
        #print("shifted:",shifted)
        byte |= shifted >> offset;
        #print("byte",byte)
        bits_left -= to_fill
        #print("bits_left:",bits_left)
        next_read_bit += to_fill
        #print("next_read:",next_read_bit)
        if bits_left <= 0:

            out_bytes.append(byte)
            byte = 0
            bits_left = 8

        if next_read_bit >= chunk_bits:
            next_read_chunk += 1
            next_read_bit -= chunk_bits
   # print(out_bytes)

    return out_bytes

def decode_file(input_file, speed):
    wav = wave.open(input_file)
    if wav.getnchannels() == 2:
        mono = StringIO()
        stereo_to_mono(input_file, mono)

        mono.seek(0)
        input_file = mono
    wav.close()

    offset = 0
    for frame_rate, chunk in yield_chunks(input_file, speed / 2):
        dom = dominant(frame_rate, chunk)
        print("{} => {}".format(offset, dom))
        offset += 1

arr_chunks = []
def extract_packet(freqs):
    global arr_chunks
    freqs = freqs[::2]
    bit_chunks = [int(round((f - START_HZ) / STEP_HZ)) for f in freqs]
    bit_chunks = [c for c in bit_chunks[1:] if 0 <= c < (2 ** BITS)]
    arr_chunks = bit_chunks[18:]
    print("arr_chunks",arr_chunks)
    return bytearray(decode_bitchunks(BITS, bit_chunks))

def display(s):
    cprint(figlet_format(s.replace(' ', '   '), font='doom'), 'yellow')

def write_linux():
    print("---a---")
    fs=44100
    ts=0.9  
    t=np.arange(fs*ts)
    sample = []
    sample.append(HANDSHAKE_START_HZ)
    print("arr_chunks2",arr_chunks)
    for i in range(0,len(arr_chunks)):
        sample.append((arr_chunks[i] >> 4)*STEP_HZ+START_HZ)
        sample.append((arr_chunks[i] & 0xf)*STEP_HZ+START_HZ)
    sample.append(HANDSHAKE_END_HZ)
    print(sample)
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paFloat32,channels=1,rate=44100,output=True)
    for i in range(0,len(sample)):
        samples = np.sin(2*np.pi*sample[i]/44100*t).astype(np.float32)
        stream.write(samples)

def listen_linux(frame_rate=44100, interval=0.1):
    mic = alsaaudio.PCM(alsaaudio.PCM_CAPTURE, alsaaudio.PCM_NORMAL)
    mic.setchannels(1)
    mic.setrate(44100)
    mic.setformat(alsaaudio.PCM_FORMAT_S16_LE)

    num_frames = int(round((interval / 2) * frame_rate))
    mic.setperiodsize(num_frames)
    print("start...")

    in_packet = False
    packet = []

    while True:
        l, data = mic.read()
        if not l:
            continue

        chunk = np.fromstring(data, dtype=np.int16)
        dom = dominant(frame_rate, chunk)

        if in_packet and match(dom, HANDSHAKE_END_HZ):
            byte_stream = extract_packet(packet)
            print("original code",byte_stream)

            try:
                byte_stream = RSCodec(FEC_BYTES).decode(byte_stream)
                byte_stream = byte_stream.decode("utf-8")
                stu_num = byte_stream[:9]
                print("stu_num",stu_num)
                #byte_stream = byte_stream[9:]
                #print("byte_stream",byte_stream)
                if int(stu_num)==201602086 :
                    display(byte_stream[9:])
                    display("")
                    write_linux()
            except ReedSolomonError as e:
                print("{}: {}".format(e, byte_stream))

            packet = []
            in_packet = False
        elif in_packet:
            packet.append(dom)
        elif match(dom, HANDSHAKE_START_HZ):
            in_packet = True

if __name__ == '__main__':
    colorama.init(strip=not sys.stdout.isatty())

    #decode_file(sys.argv[1], float(sys.argv[2]))
    listen_linux()
