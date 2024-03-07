import subprocess
import re
import os




def ffmpeg_extract_audio(video_input, audio_output, start_time, end_time):
    
    command = ['ffmpeg', '-ss',str(start_time), '-to', str(end_time), '-i', f'{video_input}', "-vn",
                   '-c:a', 'pcm_s16le','-y', audio_output, '-loglevel', 'quiet']
    
    subprocess.run(command)

def make_filename_safe(filename):
    filename = re.sub(r'[\\/:*?"<>|_]', '', filename)
    filename = re.sub(r'\s+', ' ', filename)
    filename = filename.strip()
    return filename

def get_subdir(folder_path):
    subdirectories = [os.path.abspath(os.path.join(folder_path, name)) for name in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, name))]
    return subdirectories

def get_filename(directory,format=None):
    file_list = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            if not file.startswith('.') and os.path.isfile(file_path):
                if format:
                    if file.endswith(format):
                        file_list.append([file,file_path])
                else:
                    file_list.append([file, file_path])
    file_list.sort()
    return file_list

def srt_format_timestamp( seconds):
        assert seconds >= 0, "non-negative timestamp expected"
        milliseconds = round(seconds * 1000.0)

        hours = milliseconds // 3_600_000
        milliseconds -= hours * 3_600_000

        minutes = milliseconds // 60_000
        milliseconds -= minutes * 60_000

        seconds = milliseconds // 1_000
        milliseconds -= seconds * 1_000

        return (f"{hours:02d}:") + f"{minutes:02d}:{seconds:02d}.{milliseconds:03d}"
    