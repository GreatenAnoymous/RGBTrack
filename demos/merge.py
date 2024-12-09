from moviepy.editor import VideoFileClip, TextClip, CompositeVideoClip, clips_array

# File paths for the input videos
video_path_unimproved = "fp_tracking_unimproved.mp4"
video_path_improved = "fp_tracking_improved.mp4"

# Load the video clips
clip_unimproved = VideoFileClip(video_path_unimproved)
clip_improved = VideoFileClip(video_path_improved)
min_duration = min(clip_unimproved.duration, clip_improved.duration)
# Ensure both videos have the same duration
min_duration = min(clip_unimproved.duration, clip_improved.duration)
clip_unimproved = clip_unimproved.subclip(0, min_duration)
clip_improved = clip_improved.subclip(0, min_duration)

# Add titles to the videos
title_unimproved = TextClip("Baseline", fontsize=50, color='white', bg_color='black', size=(clip_unimproved.w, 80))
title_improved = TextClip("Ours", fontsize=50, color='white', bg_color='black', size=(clip_improved.w, 80))

# Position the titles above the respective videos
clip_unimproved_with_title = CompositeVideoClip([clip_unimproved.set_position(('center', 'top')), 
                                                 title_unimproved.set_position(('center', 'top'))], 
                                                 size=(clip_unimproved.w, clip_unimproved.h + 80))
clip_improved_with_title = CompositeVideoClip([clip_improved.set_position(('center', 'top')), 
                                               title_improved.set_position(('center', 'top'))], 
                                               size=(clip_improved.w, clip_improved.h + 80))

# Combine the clips side by side
final_clip = clips_array([[clip_unimproved_with_title, clip_improved_with_title]])
final_clip = final_clip.set_duration(min_duration)
# Write the result to a file
output_path = "merged_video.mp4"
final_clip.write_videofile(output_path, codec="libx264", fps=24)
