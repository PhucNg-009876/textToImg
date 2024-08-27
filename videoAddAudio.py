from moviepy.editor import *

# Đường dẫn tới file video và audio
video_file = "video.mp4"
audio_file = "videoplayback.mp3"

# Đọc video và audio
video_clip = VideoFileClip(video_file)
audio_clip = AudioFileClip(audio_file)

# Ghép audio vào video
final_clip = video_clip.set_audio(audio_clip)

# Lưu video mới với audio đã ghép
final_clip.write_videofile("output_video.mp4", codec="libx264", audio_codec="aac")

# Đóng các clip
video_clip.close()
audio_clip.close()
