import os
import sys


def get_video_path():
    captures_dir = 'captures'
    video_path = None
    if len(sys.argv) >= 2:
        # If argument is given, use it (search in captures/ if not absolute)
        arg_path = sys.argv[1]
        if not os.path.isabs(arg_path):
            candidate = os.path.join(captures_dir, arg_path)
            if os.path.exists(candidate):
                video_path = candidate
            elif os.path.exists(arg_path):
                video_path = arg_path
            else:
                print(f"Video file not found: {arg_path}")
                return None
        else:
            if os.path.exists(arg_path):
                video_path = arg_path
            else:
                print(f"Video file not found: {arg_path}")
                return None
    else:
        # No argument: use latest file in captures/
        if not os.path.exists(captures_dir):
            print("No captures/ directory found. Please record a video first.")
            return None
        files = [f for f in os.listdir(captures_dir) if f.lower().endswith(('.avi', '.mp4', '.mov'))]
        if not files:
            print("No video files found in captures/. Please record a video first.")
            return None
        files.sort(key=lambda f: os.path.getmtime(os.path.join(captures_dir, f)), reverse=True)
        video_path = os.path.join(captures_dir, files[0])
        print(f"No video argument given. Using latest capture: {files[0]}")
    return video_path