import subprocess
import re
import os




def ffmpeg_extract_audio(video_input, audio_output, start_time, end_time):
    """Extracts the audio clip from video file

    Args:
        video_input (str): path of the video
        audio_output (str): path of the audio file to create
        start_time (str): start time of the clip
        end_time (str): end time of the clip
    """
    command = ['ffmpeg', '-ss',str(start_time), '-to', str(end_time), '-i', f'{video_input}', "-vn",
                   '-c:a', 'pcm_s16le','-y', audio_output, '-loglevel', 'quiet']
    
    subprocess.run(command)
    
def ffmpeg_video_2_audio(video_input, audio_output):
    
    command = ['ffmpeg', '-i', f'{video_input}', "-q:a", "0",
                   '-map', '0', audio_output, '-loglevel', 'quiet']
    
    subprocess.run(command)
    

def make_filename_safe(filename:str) -> str:
    """Creates a filename that does not contain problematic symbols

    Args:
        filename (str): _description_

    Returns:
        str: _description_
    """
    filename = re.sub(r'[\\/:*?"<>|_]', '', filename)
    filename = re.sub(r'\s+', ' ', filename)
    filename = filename.strip()
    return filename

def get_subdir(folder_path:str) -> list[str]:
    """Gets a list of the directories inside the folder path directory

    Args:
        folder_path (str): path to look inside

    Returns:
        list[str]: list of directories inside the folder path
    """
    subdirectories = [os.path.abspath(os.path.join(folder_path, name)) for name in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, name))]
    return subdirectories

def get_filename(directory:str,format:str=None) -> list[str]:
    """Return all the files inside the directory given the desired format

    Args:
        directory (str): _description_
        format (str, optional): _description_. Defaults to None.

    Returns:
        list[str]: _description_
    """
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
    