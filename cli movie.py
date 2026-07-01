#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
视频转终端 ANSI 动画 + 高保真多频 PCM 合成音频（播放端低通滤波降噪 + 动态范围控制）
转换器在 Windows 上运行，生成播放器脚本。
"""

import os
import sys
import cv2
import numpy as np
import base64
import gzip
import subprocess
import json
from pathlib import Path

# ------------------ 用户交互 ------------------
def get_mp4_files():
    return sorted([f for f in os.listdir('.') if f.lower().endswith('.mp4')])

def choose_video(files):
    print("请选择视频文件：")
    for i, f in enumerate(files, 1):
        print(f"  {i}. {f}")
    while True:
        try:
            choice = int(input("请输入编号: "))
            if 1 <= choice <= len(files):
                return files[choice-1]
        except:
            pass
        print(f"请输入 1~{len(files)} 之间的数字")

def get_width():
    w = input("请输入存储宽度（字符数，推荐100~160，默认120）: ").strip()
    if w == "":
        return 120
    try:
        w = int(w)
        return max(40, min(300, w))
    except:
        return 120

# ------------------ 高级音频合成参数 ------------------
SAMPLE_RATE = 44100
HARMONIC_COUNT = 5
FRAME_DURATION_MS = 40
MIN_FREQ = 40
MAX_FREQ = 8000
CROSSFADE_MS = 8
NOISE_THRESHOLD = 0.15      # 只保留幅度 > 最大幅度 * 0.15 的频率
VOLUME = 0.98               # 转换器端合成时音量（避免预失真）
LOWPASS_ALPHA = 0.3

# ------------------ 音频提取 + 多频分析 + 降噪合成 ------------------
def extract_audio_pcm(video_path, sample_rate):
    cmd = [
        'ffmpeg', '-i', video_path,
        '-f', 's16le',
        '-acodec', 'pcm_s16le',
        '-ac', '1',
        '-ar', str(sample_rate),
        '-vn',
        '-'
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, check=True)
        return result.stdout, sample_rate
    except subprocess.CalledProcessError as e:
        print(f"音频提取失败: {e.stderr.decode()}")
        return None, None

def soft_limit(x, limit=0.98):
    return np.tanh(x / limit) * limit

def lowpass_filter(signal, alpha=0.3):
    """一阶低通滤波"""
    filtered = np.zeros_like(signal)
    filtered[0] = signal[0]
    for i in range(1, len(signal)):
        filtered[i] = alpha * signal[i] + (1 - alpha) * filtered[i-1]
    return filtered

def pcm_to_multifreq_pcm(pcm_data, sample_rate, frame_duration_ms=FRAME_DURATION_MS,
                         harmonic_count=HARMONIC_COUNT, crossfade_ms=CROSSFADE_MS):
    samples_per_frame = int(sample_rate * frame_duration_ms / 1000.0)
    if samples_per_frame < 2:
        samples_per_frame = 2
    data = np.frombuffer(pcm_data, dtype=np.int16).astype(np.float32) / 32768.0
    total_samples = len(data)
    segments = []
    prev_freqs = None
    prev_phases = None

    for start in range(0, total_samples, samples_per_frame):
        end = min(start + samples_per_frame, total_samples)
        chunk = data[start:end]
        if len(chunk) < 2:
            break
        window = np.hanning(len(chunk))
        windowed = chunk * window
        fft = np.fft.rfft(windowed)
        magnitude = np.abs(fft)
        freqs = np.fft.rfftfreq(len(windowed), 1/sample_rate)

        valid = (freqs >= MIN_FREQ) & (freqs <= MAX_FREQ)
        if not np.any(valid):
            seg = np.zeros(samples_per_frame, dtype=np.float32)
            prev_freqs = None
            prev_phases = None
            segments.append(seg)
            continue

        mag_valid = magnitude[valid]
        freq_valid = freqs[valid]
        max_mag = np.max(mag_valid)
        threshold = max_mag * NOISE_THRESHOLD
        keep = mag_valid > threshold
        if not np.any(keep):
            seg = np.zeros(samples_per_frame, dtype=np.float32)
            segments.append(seg)
            continue

        mag_valid = mag_valid[keep]
        freq_valid = freq_valid[keep]
        idx_sorted = np.argsort(mag_valid)[::-1][:harmonic_count]
        current_freqs = freq_valid[idx_sorted]
        amp_weights = mag_valid[idx_sorted] / (np.sum(mag_valid[idx_sorted]) + 1e-6)

        if prev_freqs is None:
            phases = np.random.uniform(0, 2*np.pi, size=len(current_freqs))
        else:
            phases = np.zeros(len(current_freqs))
            for i, cf in enumerate(current_freqs):
                if len(prev_freqs) == 0:
                    phases[i] = np.random.uniform(0, 2*np.pi)
                else:
                    idx = np.argmin(np.abs(prev_freqs - cf))
                    delta_phase = 2 * np.pi * (cf - prev_freqs[idx]) * (frame_duration_ms / 1000.0)
                    phases[i] = prev_phases[idx] + delta_phase
                    phases[i] %= (2*np.pi)

        t = np.arange(samples_per_frame) / sample_rate
        y = np.zeros(samples_per_frame, dtype=np.float32)
        for f, p, w in zip(current_freqs, phases, amp_weights):
            y += w * np.sin(2*np.pi*f*t + p)

        max_amp = np.max(np.abs(y))
        if max_amp > 0:
            y = y / max_amp * VOLUME
        y = soft_limit(y, limit=0.99)

        if len(segments) > 0 and crossfade_ms > 0:
            crossfade_samples = int(sample_rate * crossfade_ms / 1000.0)
            if crossfade_samples > 0 and crossfade_samples < len(y):
                prev_seg = segments[-1]
                if len(prev_seg) >= crossfade_samples:
                    fade_out = np.linspace(1.0, 0.0, crossfade_samples)
                    fade_in = np.linspace(0.0, 1.0, crossfade_samples)
                    y[:crossfade_samples] = y[:crossfade_samples] * fade_in + prev_seg[-crossfade_samples:] * fade_out

        y = lowpass_filter(y, LOWPASS_ALPHA)
        segments.append(y)
        prev_freqs = current_freqs.copy()
        prev_phases = phases.copy()

    all_float = np.concatenate(segments)
    all_int16 = (all_float * 32767).clip(-32768, 32767).astype(np.int16)
    return all_int16.tobytes()

# ------------------ 主转换函数 ------------------
def video_to_pcm_script(video_path, output_script_path, target_fps=30, store_width=120):
    # 视频帧处理
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return False, f"无法打开视频: {video_path}"
    orig_fps = cap.get(cv2.CAP_PROP_FPS)
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    store_h = max(1, int(store_width * orig_h / orig_w))
    store_w = store_width
    step = 1 if orig_fps <= target_fps else int(round(orig_fps / target_fps))
    eff_fps = orig_fps / step
    print(f"原视频: {orig_w}x{orig_h}, {orig_fps:.2f}fps")
    print(f"存储尺寸: {store_w} x {store_h}")
    print(f"实际帧率: {eff_fps:.2f}fps")
    print("提取视频帧中...")
    frames = []
    frame_idx = 0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % step == 0:
            small = cv2.resize(frame, (store_w, store_h), interpolation=cv2.INTER_LANCZOS4)
            rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
            frames.append(rgb)
        frame_idx += 1
        if frame_idx % 100 == 0:
            print(f"  进度: {frame_idx}/{total} 帧", end='\r')
    cap.release()
    print(f"\n实际帧数: {len(frames)}")
    frames_arr = np.array(frames, dtype=np.uint8)
    compressed_video = gzip.compress(frames_arr.tobytes())
    b64_video = base64.b64encode(compressed_video).decode('ascii')
    print(f"视频压缩: {len(compressed_video)/1024:.1f} KB")

    # 音频处理
    print(f"提取音频并合成多频 PCM (采样率 {SAMPLE_RATE}Hz)...")
    pcm_data, _ = extract_audio_pcm(video_path, SAMPLE_RATE)
    if pcm_data is None:
        print("音频提取失败，将生成无音频版本")
        b64_audio = ""
        b64_meta = ""
        audio_nonzero = False
    else:
        all_pcm = pcm_to_multifreq_pcm(pcm_data, SAMPLE_RATE)
        audio_nonzero = any(b != 0 for b in all_pcm)
        compressed_audio = gzip.compress(all_pcm)
        b64_audio = base64.b64encode(compressed_audio).decode('ascii')
        print(f"音频 PCM 压缩: {len(compressed_audio)/1024:.1f} KB")
        audio_meta = {
            'sample_rate': SAMPLE_RATE,
            'total_bytes': len(all_pcm),
            'non_zero': audio_nonzero
        }
        meta_json = json.dumps(audio_meta)
        b64_meta = base64.b64encode(meta_json.encode()).decode('ascii')

    # 生成播放器脚本（包含低通滤波降噪）
    with open(output_script_path, 'w', encoding='utf-8') as f:
        f.write('#!/usr/bin/env python3\n')
        f.write('# -*- coding: utf-8 -*-\n')
        f.write('"""终端视频播放器 + 高保真多频 PCM 合成音频（低通滤波降噪 + 动态控制）"""\n')
        f.write('import sys, time, base64, gzip, numpy as np\n')
        f.write('import subprocess, threading, json, tempfile, os\n\n')
        f.write('def lowpass_filter(signal, alpha=0.3):\n')
        f.write('    """一阶低通滤波，平滑高频噪声"""\n')
        f.write('    filtered = np.zeros_like(signal)\n')
        f.write('    filtered[0] = signal[0]\n')
        f.write('    for i in range(1, len(signal)):\n')
        f.write('        filtered[i] = alpha * signal[i] + (1 - alpha) * filtered[i-1]\n')
        f.write('    return filtered\n\n')
        f.write('def normalize_and_limit(pcm_data, target_peak=0.95, gain_boost=1.0, smooth=0.3):\n')
        f.write('    """归一化 + 软限幅 + 低通滤波"""\n')
        f.write('    samples = np.frombuffer(pcm_data, dtype=np.int16).astype(np.float32) / 32768.0\n')
        f.write('    samples = samples * gain_boost\n')
        f.write('    max_abs = np.max(np.abs(samples))\n')
        f.write('    if max_abs > target_peak:\n')
        f.write('        gain = target_peak / max_abs\n')
        f.write('        samples = samples * gain\n')
        f.write('    # 低通滤波去噪\n')
        f.write('    if smooth > 0:\n')
        f.write('        samples = lowpass_filter(samples, alpha=smooth)\n')
        f.write('    samples = (samples * 32767).clip(-32768, 32767).astype(np.int16)\n')
        f.write('    return samples.tobytes()\n\n')
        f.write('def play_audio_pcm(pcm_data, sample_rate, device=None, debug=False, volume=1.0, smooth=0.3):\n')
        f.write('    pcm_data = normalize_and_limit(pcm_data, target_peak=0.95, gain_boost=volume, smooth=smooth)\n')
        f.write('    if subprocess.run(["which", "aplay"], capture_output=True).returncode == 0:\n')
        f.write('        cmd = ["aplay", "-q", "-f", "S16_LE", "-c", "1", "-r", str(sample_rate), "-t", "raw"]\n')
        f.write('        if device: cmd += ["-D", device]\n')
        f.write('        return _play_pipe(cmd, pcm_data, debug)\n')
        f.write('    if subprocess.run(["which", "paplay"], capture_output=True).returncode == 0:\n')
        f.write('        cmd = ["paplay", "--raw", "--format=s16le", "--channels=1", "--rate="+str(sample_rate)]\n')
        f.write('        return _play_pipe(cmd, pcm_data, debug)\n')
        f.write('    if subprocess.run(["which", "ffplay"], capture_output=True).returncode == 0:\n')
        f.write('        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:\n')
        f.write('            import wave\n')
        f.write('            with wave.open(tmp.name, "wb") as wav:\n')
        f.write('                wav.setnchannels(1)\n')
        f.write('                wav.setsampwidth(2)\n')
        f.write('                wav.setframerate(sample_rate)\n')
        f.write('                wav.writeframes(pcm_data)\n')
        f.write('            cmd = ["ffplay", "-nodisp", "-autoexit", tmp.name]\n')
        f.write('            ret = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode\n')
        f.write('            os.unlink(tmp.name)\n')
        f.write('            return ret == 0\n')
        f.write('    print("未找到可用的音频播放器", file=sys.stderr)\n')
        f.write('    return False\n\n')
        f.write('def _play_pipe(cmd, pcm_data, debug):\n')
        f.write('    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)\n')
        f.write('    proc.stdin.write(pcm_data)\n')
        f.write('    proc.stdin.close()\n')
        f.write('    ret = proc.wait()\n')
        f.write('    if ret != 0 and debug:\n')
        f.write('        print(f"命令 {cmd[0]} 失败，返回码 {ret}", file=sys.stderr)\n')
        f.write('    return ret == 0\n\n')
        f.write('def get_term_size():\n')
        f.write('    try:\n')
        f.write('        import shutil\n')
        f.write('        return shutil.get_terminal_size()\n')
        f.write('    except:\n')
        f.write('        return (80, 24)\n\n')
        f.write('def main():\n')
        f.write('    import argparse\n')
        f.write('    parser = argparse.ArgumentParser()\n')
        f.write('    parser.add_argument("--fps", type=float, help="播放帧率")\n')
        f.write('    parser.add_argument("--no-block", action="store_true", help="禁用半块字符")\n')
        f.write('    parser.add_argument("--no-audio", action="store_true", help="禁用音频")\n')
        f.write('    parser.add_argument("--debug", action="store_true", help="调试模式")\n')
        f.write('    parser.add_argument("--audio-device", type=str, help="指定音频设备（如 hw:0,0）")\n')
        f.write('    parser.add_argument("--volume", type=float, default=1.0, help="音量增益 (0.5~2.0, 默认1.0)")\n')
        f.write('    parser.add_argument("--smooth", type=float, default=0.3, help="低通滤波强度 (0~1, 0=无滤波, 1=极平滑, 默认0.3)")\n')
        f.write('    args = parser.parse_args()\n\n')
        f.write(f'    b64_video = """{b64_video}"""\n')
        f.write('    compressed_video = base64.b64decode(b64_video)\n')
        f.write('    data_video = gzip.decompress(compressed_video)\n')
        f.write(f'    frames = np.frombuffer(data_video, dtype=np.uint8).reshape({len(frames)}, {store_h}, {store_w}, 3)\n')
        f.write(f'    stored_fps = {eff_fps:.2f}\n')
        f.write('    fps = args.fps if args.fps else stored_fps\n')
        f.write('    delay = 1.0 / fps\n\n')
        if pcm_data is not None and audio_nonzero:
            f.write(f'    b64_audio = """{b64_audio}"""\n')
            f.write(f'    b64_meta = """{b64_meta}"""\n')
            f.write('    if not args.no_audio:\n')
            f.write('        compressed_audio = base64.b64decode(b64_audio)\n')
            f.write('        all_pcm = gzip.decompress(compressed_audio)\n')
            f.write('        meta = json.loads(base64.b64decode(b64_meta).decode())\n')
            f.write('        sample_rate = meta["sample_rate"]\n')
            f.write('        if meta["non_zero"]:\n')
            f.write('            def audio():\n')
            f.write('                ok = play_audio_pcm(all_pcm, sample_rate, args.audio_device, args.debug, args.volume, args.smooth)\n')
            f.write('                if not ok and args.debug:\n')
            f.write('                    print("音频播放失败，请检查声卡或安装 alsa-utils/pulseaudio-utils", file=sys.stderr)\n')
            f.write('            threading.Thread(target=audio, daemon=True).start()\n')
            f.write('        elif args.debug:\n')
            f.write('            print("音频数据全为零，跳过播放", file=sys.stderr)\n')
        f.write('\n    term_w, term_h = get_term_size()\n')
        f.write('    h, w = frames.shape[1], frames.shape[2]\n')
        f.write('    if args.no_block:\n')
        f.write('        disp_h = term_h\n')
        f.write('    else:\n')
        f.write('        disp_h = term_h * 2\n')
        f.write('    disp_w = term_w\n')
        f.write('    scale = min(disp_w / w, disp_h / h)\n')
        f.write('    target_w = max(1, int(w * scale))\n')
        f.write('    target_h = max(1, int(h * scale))\n')
        f.write('    pad_left = max(0, (term_w - target_w) // 2)\n\n')
        f.write('    sys.stdout.write("\\033[2J\\033[?25l\\033[H")\n')
        f.write('    sys.stdout.flush()\n\n')
        f.write('    try:\n')
        f.write('        for frame in frames:\n')
        f.write('            start = time.perf_counter()\n')
        f.write('            if frame.shape[0] != target_h or frame.shape[1] != target_w:\n')
        f.write('                h0, w0 = frame.shape[:2]\n')
        f.write('                y_idx = (np.arange(target_h) * h0 / target_h).astype(int)\n')
        f.write('                x_idx = (np.arange(target_w) * w0 / target_w).astype(int)\n')
        f.write('                disp = frame[y_idx][:, x_idx]\n')
        f.write('            else:\n')
        f.write('                disp = frame\n')
        f.write('            lines = []\n')
        f.write('            if args.no_block:\n')
        f.write('                for y in range(target_h):\n')
        f.write('                    line = " " * pad_left\n')
        f.write('                    for x in range(target_w):\n')
        f.write('                        r, g, b = disp[y, x]\n')
        f.write('                        line += f"\\033[48;2;{r};{g};{b}m "\n')
        f.write('                    lines.append(line + "\\033[0m")\n')
        f.write('            else:\n')
        f.write('                for y in range(0, target_h, 2):\n')
        f.write('                    line = " " * pad_left\n')
        f.write('                    for x in range(target_w):\n')
        f.write('                        r1, g1, b1 = disp[y, x]\n')
        f.write('                        if y+1 < target_h:\n')
        f.write('                            r2, g2, b2 = disp[y+1, x]\n')
        f.write('                            line += f"\\033[38;2;{r2};{g2};{b2}m\\033[48;2;{r1};{g1};{b1}m▀"\n')
        f.write('                        else:\n')
        f.write('                            line += f"\\033[48;2;{r1};{g1};{b1}m \\033[0m"\n')
        f.write('                    lines.append(line + "\\033[0m")\n')
        f.write('            sys.stdout.write("\\033[H" + "\\n".join(lines) + "\\n")\n')
        f.write('            sys.stdout.flush()\n')
        f.write('            elapsed = time.perf_counter() - start\n')
        f.write('            if elapsed < delay:\n')
        f.write('                time.sleep(delay - elapsed)\n')
        f.write('    except KeyboardInterrupt:\n')
        f.write('        pass\n')
        f.write('    finally:\n')
        f.write('        sys.stdout.write("\\033[?25h\\033[0m\\n")\n')
        f.write('        sys.stdout.flush()\n\n')
        f.write('if __name__ == "__main__":\n')
        f.write('    main()\n')
    
    return True, f"生成成功: {output_script_path} (视频尺寸 {store_w}x{store_h}, 音频长度 {len(all_pcm) if pcm_data else 0} bytes)"

def main():
    try:
        subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
    except:
        print("错误: 需要 ffmpeg，请安装并添加到 PATH")
        sys.exit(1)
    mp4_files = get_mp4_files()
    if not mp4_files:
        print("当前目录没有 .mp4 文件")
        sys.exit(1)
    selected = choose_video(mp4_files)
    width = get_width()
    base = Path(selected).stem
    out_name = f"{base}(cil动画多频).txt"
    out_path = os.path.join(os.getcwd(), out_name)
    print(f"\n处理: {selected} -> {out_name}")
    ok, msg = video_to_pcm_script(selected, out_path, target_fps=30, store_width=width)
    if ok:
        print(msg)
        print(f"\n将 {out_name} 复制到 Linux 终端，运行: python '{out_name}' [--volume 1.2] [--smooth 0.4] [--audio-device hw:0,0]")
        print("--smooth 可调节降噪强度（0=无滤波, 1=极平滑），默认0.3")
    else:
        print("错误:", msg)

if __name__ == "__main__":
    try:
        import cv2, numpy
    except ImportError:
        print("请先安装依赖: pip install opencv-python numpy")
        sys.exit(1)
    main()