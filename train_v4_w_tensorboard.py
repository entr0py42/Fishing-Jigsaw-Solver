import argparse
import os
import subprocess
import sys
import time
import webbrowser

LOG_DIR = "tb_logs"
PORT = 6006
DASHBOARD_URL = f"http://localhost:{PORT}/"
TRAIN_SCRIPT = "train_v4.py"


def start_tensorboard() -> subprocess.Popen:
    os.makedirs(LOG_DIR, exist_ok=True)
    print(f"--- Starting TensorBoard on {DASHBOARD_URL} ---")

    server = subprocess.Popen(
        [sys.executable, "-m", "tensorboard.main", f"--logdir={LOG_DIR}", f"--port={PORT}"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    time.sleep(2)
    webbrowser.open(DASHBOARD_URL)
    return server


def parse_args():
    parser = argparse.ArgumentParser(description="Train with a live TensorBoard dashboard.")
    parser.add_argument(
        "--cuda",
        type=int,
        default=0,
        choices=[0, 1],
        help="Set to 1 to train on the GPU (CUDA). Defaults to 0 (CPU).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    tensorboard = start_tensorboard()

    print(f"--- Launching Training Script ({TRAIN_SCRIPT}) ---\n")
    try:
        subprocess.run([sys.executable, TRAIN_SCRIPT, "--cuda", str(args.cuda)], check=True)
    except KeyboardInterrupt:
        print("\n--- Training manually interrupted by user ---")
    except subprocess.CalledProcessError as error:
        print(f"\n--- Training script failed with exit code {error.returncode} ---")
    finally:
        print("--- Terminating TensorBoard server ---")
        tensorboard.terminate()
        tensorboard.wait()


if __name__ == "__main__":
    main()
