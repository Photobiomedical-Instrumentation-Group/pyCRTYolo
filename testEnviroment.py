
# testEnviroment.py
# This script checks the environment setup for the pyCRTYolo project.

import importlib
import subprocess
import sys
import os
import urllib.request
import zipfile
import shutil

# -------- Função para verificar FFmpeg --------
def is_ffmpeg_installed():
    print("\n🧰 Checking if FFmpeg is installed...")

    # Tenta executar ffmpeg direto (se estiver no PATH)
    try:
        result = subprocess.run(['ffmpeg', '-version'], capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ FFmpeg is installed.")
            print(f"📌 Version: {result.stdout.splitlines()[0]}")
            return True
    except FileNotFoundError:
        pass

    # Tenta localizar ffmpeg.exe em local comum, exemplo C:\ffmpeg\bin
    ffmpeg_path = r"C:\ffmpeg\bin\ffmpeg.exe"
    if os.path.exists(ffmpeg_path):
        try:
            result = subprocess.run([ffmpeg_path, '-version'], capture_output=True, text=True)
            if result.returncode == 0:
                print("✅ FFmpeg found in C:\\ffmpeg\\bin.")
                print(f"📌 Version: {result.stdout.splitlines()[0]}")
                return True
        except Exception:
            pass

    print("❌ FFmpeg is not installed or not in system PATH.")
    return False

def download_and_install_ffmpeg():
    ffmpeg_url = "https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-win64-gpl.zip"
    zip_path = "ffmpeg.zip"
    install_dir = "C:\\ffmpeg"

    try:
        print("\n⬇️ Downloading FFmpeg...")
        urllib.request.urlretrieve(ffmpeg_url, zip_path)
        print("✅ Download complete.")

        print("\n🗜️ Extracting FFmpeg...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(".")
        print("✅ Extraction complete.")

        extracted_folders = [f for f in os.listdir(".") if f.startswith("ffmpeg-master-latest-win64-gpl")]
        if not extracted_folders:
            print("❌ Could not find extracted FFmpeg folder.")
            sys.exit(1)

        extracted_folder = extracted_folders[0]

        if os.path.exists(install_dir):
            print(f"\n🗑️ Removing old installation at {install_dir} ...")
            shutil.rmtree(install_dir)

        print(f"\n📁 Moving FFmpeg folder to {install_dir} ...")
        os.rename(extracted_folder, install_dir)
        print("✅ Move complete.")

        print("\n🛠️ Adding FFmpeg bin folder to system PATH...")
        bin_path = os.path.join(install_dir, "bin")
        current_path = os.environ.get("PATH", "")
        if bin_path.lower() not in current_path.lower():
            subprocess.run(f'setx PATH "%PATH%;{bin_path}"', shell=True)
            print(f"✅ PATH updated with {bin_path}. Please restart your terminal or PC to apply changes.")
        else:
            print("ℹ️ FFmpeg bin folder is already in PATH.")

        if os.path.exists(zip_path):
            os.remove(zip_path)
            print("\n🧹 Cleaned up downloaded zip file.")

        print("\n🎉 FFmpeg installation completed successfully!")

    except Exception as e:
        print(f"❌ Error during FFmpeg installation: {e}")
        sys.exit(1)


# ====== 1. DEPENDENCY CHECK ======
required_packages = [
    'cv2', 'numpy', 'matplotlib', 'torch', 'decord', 'PIL',
    'tkinter', 'pandas'
]

local_pyCRT_path = "C:/Users/raque/OneDrive/Documentos/GitHub/pyCRT"

try:
    if os.path.exists(local_pyCRT_path):
        if local_pyCRT_path not in sys.path:
            sys.path.append(local_pyCRT_path)
        from src.pyCRT import PCRT
        print("✅ pyCRT imported from local path.")
    else:
        import pyCRT
        print("✅ pyCRT imported from installed package.")
except ImportError:
    print("❌ pyCRT not found locally or installed.")
    print("👉 Please install it with 'pip install pyCRT-dev' or clone it locally.")
    sys.exit(1)
    


missing_packages = []
for pkg in required_packages:
    try:
        if pkg == 'cv2':
            import cv2
        elif pkg == 'PIL':
            from PIL import Image
        elif pkg == 'tkinter':
            import tkinter
        else:
            importlib.import_module(pkg)
    except ImportError:
        missing_packages.append(pkg)

if missing_packages:
    print("\n🚨 Missing required packages:")
    for pkg in missing_packages:
        print(f" - {pkg}")
    print("\nPlease install them with:")
    pip_names = [p if p != 'cv2' else 'opencv-python' for p in missing_packages]
    print(f"pip install {' '.join(pip_names)}")
    sys.exit(1)
else:
    print("✅ All required packages are installed.")


# ====== 2. CHECK AND INSTALL FFMPEG ======
if not is_ffmpeg_installed():
    user_input = input("Would you like to download and install FFmpeg automatically? (y/n): ").strip().lower()
    if user_input == 'y':
        download_and_install_ffmpeg()
    else:
        print("⚠️ FFmpeg installation skipped. The program may not work correctly without it.")
        sys.exit(1)
else:
    print("✅ FFmpeg check passed.")


# ====== 3. MODULE IMPORT CHECK ======
print("\n📦 Checking module imports...")

modules = [
    'processLucasKanade',
    'processYolo',
    'validationROI',
]

for module in modules:
    try:
        importlib.import_module(module)
        print(f"✅ {module} imported successfully.")
    except Exception as e:
        print(f"❌ Failed to import {module}: {e}")
        sys.exit(1)


# ====== 4. UNIT TESTS ======
print("\n🧪 Running unit tests...")

try:
    from processLucasKanade import calculateMovement
    import numpy as np

    oldPts = np.array([[0, 0], [1, 1]])
    newPts = np.array([[1, 1], [2, 2]])
    mov = calculateMovement(oldPts, newPts)
    assert round(mov, 2) == 1.41, f"Unexpected result: {mov}"
    print("✅ calculateMovement passed.")
except Exception as e:
    print(f"❌ Error in calculateMovement: {e}")

try:
    from processYolo import detectFinger
    import numpy as np
    import cv2

    dummy_img = np.zeros((640, 480, 3), dtype=np.uint8)
    detectFinger(dummy_img, 0.25)
    print("✅ detectFinger ran with dummy image.")
except Exception as e:
    print(f"❌ detectFinger failed: {e}")

try:
    from validationROI import ROIValidator
    dummy_rois = [(100, 100, 50, 50)]
    dummy_video = "test.mp4"
    dummy_frame = 0

    if not os.path.exists(dummy_video):
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(dummy_video, fourcc, 1, (320, 240))
        for _ in range(5):
            out.write(np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8))
        out.release()

    print("✅ Dummy video generated for ROIValidator test.")
except Exception as e:
    print(f"❌ Error preparing ROIValidator test: {e}")


# ====== 5. CHECK FOR VIDEO IN 'Videos/' ======
print("\n🎥 Looking for a video in the 'Videos/' directory...")

video_folder = "Videos"
video_extensions = ('.mp4', '.avi', '.mov', '.mkv')

if not os.path.exists(video_folder):
    print(f"❌ Folder '{video_folder}' does not exist.")
    sys.exit(1)

video_files = [f for f in os.listdir(video_folder) if f.lower().endswith(video_extensions)]

if not video_files:
    print("❌ No video files found in 'Videos/' folder.")
    sys.exit(1)
else:
    print(f"✅ Found {len(video_files)} video file(s):")
    for f in video_files:
        print(f" - {f}")

print("\n✅ Environment check complete. You're good to go! 🚀")
