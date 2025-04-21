import streamlit as st
import torch
import os
import sys
import zipfile
import gdown
import time
import shutil
import contextlib
from .utils import DEVICE, IMSIZE, tensor_to_pil # Import necessary utils

# Define constants within this module
STYTR_DIR = "StyTR"
STYTR_CODE_GDRIVE_ID = "11jL5m9WwF1_hDwCVlDlHYox_1qr2qX_E"
STYTR_ZIP_PATH = "StyTR.zip"
STYTR_EXPERIMENTS_DIR = os.path.join(STYTR_DIR, "experiments")

STYTR_VGG_WEIGHTS_ID = "1BinnwM5AmIcVubr16tPTqxMjUCE8iu5M"
STYTR_EMBEDDING_WEIGHTS_ID = "1C3xzTOWx8dUXXybxZwmjijZN8SrC3e4B"
STYTR_TRANSFORMER_WEIGHTS_ID = "1dnobsaLeE889T_LncCkAA2RkqzwsfHYy"

STYTR_VGG_WEIGHTS_PATH = os.path.join(STYTR_EXPERIMENTS_DIR, "vgg_normalised.pth")
STYTR_EMBEDDING_WEIGHTS_PATH = os.path.join(STYTR_EXPERIMENTS_DIR, "embedding_iter_160000.pth")
STYTR_TRANSFORMER_WEIGHTS_PATH = os.path.join(STYTR_EXPERIMENTS_DIR, "transformer_iter_160000.pth")

# Flag to ensure setup runs only once per session if needed
stytr_initialized = False
stytr_model_instance = None # To hold the loaded model
_StyTR_model_func = None
_stytr_process_images = None

# --- Context Manager for Changing Directory ---
@contextlib.contextmanager
def change_cwd(path):
    """Temporarily change the current working directory."""
    original_cwd = os.getcwd()
    try:
        os.chdir(path)
        yield
    finally:
        os.chdir(original_cwd)

# --- Helper Functions ---
def _add_stytr_to_path():
    """Adds StyTR directories to sys.path."""
    app_root = os.path.abspath(".")
    stytr_path_abs = os.path.join(app_root, STYTR_DIR)
    if app_root not in sys.path:
        sys.path.insert(0, app_root)
    if stytr_path_abs not in sys.path:
        sys.path.insert(0, stytr_path_abs)

def _import_stytr_modules():
    """Imports StyTR functions, changing CWD temporarily."""
    global _StyTR_model_func, _stytr_process_images
    if _StyTR_model_func is None or _stytr_process_images is None:
        _add_stytr_to_path() # Ensure StyTR directory is findable by Python

        stytr_dir_abs = os.path.abspath(STYTR_DIR)
        if not os.path.isdir(stytr_dir_abs):
            st.error(f"Cannot change to StyTR directory '{stytr_dir_abs}' as it doesn't exist.")
            raise FileNotFoundError(f"StyTR directory not found at {stytr_dir_abs}")

        # Use the context manager to change directory *only* during the import
        st.info(f"Temporarily changing CWD to {stytr_dir_abs} for StyTR import...")
        try:
            with change_cwd(stytr_dir_abs):
                # Inside this block, CWD is STYTR_DIR
                # Now, the import and the subsequent file loading inside utils.py should work
                from util.utils import network as loaded_network_func
                from util.utils import process_images as loaded_process_func
                _StyTR_model_func = loaded_network_func
                _stytr_process_images = loaded_process_func
            # Outside the block, CWD is automatically restored
            st.info("Changed CWD back.")
        except ImportError as e:
            st.error(f"Could not import modules from StyTR package even after changing CWD. "
                     f"Ensure '{STYTR_DIR}' contains the correct code structure (e.g., a 'util' folder with 'utils.py'). Error: {e}")
            raise
        except FileNotFoundError as e:
             # This might catch the ./experiments/vgg... error if CWD change didn't work or weights are missing
             st.error(f"FileNotFoundError during StyTR import (likely loading weights): {e}. "
                      f"Ensure weights exist in '{STYTR_EXPERIMENTS_DIR}'. Check CWD change logic.")
             raise
        except Exception as e:
            st.error(f"An unexpected error occurred during StyTR module import: {e}")
            import traceback
            st.error(traceback.format_exc())
            raise

#@st.cache_resource(show_spinner=False)
def setup_stytr():
    """Downloads StyTR code and weights if needed. Returns True if setup is successful."""
    global stytr_initialized
    if stytr_initialized:
        return True

    st.info("Checking StyTR setup...")
    try:
        if not setup_stytr_code():
            st.error("StyTR code setup failed.")
            return False

        if not setup_stytr_weights():
            st.error("StyTR weights setup failed.")
            return False

        _import_stytr_modules() # Attempt import *after* setup steps

        stytr_initialized = True
        st.success("StyTR setup complete and modules imported.")
        return True

    except ImportError:
        # Error message already shown in _import_stytr_modules
        return False
    except Exception as e:
        st.error(f"An unexpected error occurred during StyTR setup: {e}")
        import traceback
        st.error(traceback.format_exc())
        return False

def setup_stytr_code():
    """Handles downloading and setting up the StyTR code. Returns True on success."""
    if not os.path.exists(STYTR_DIR):
        st.info("StyTR code directory not found. Downloading...")
        progress_bar = st.progress(0, text="Downloading StyTR code...")
        cleanup_zip()
        temp_extract_dir = "_temp_stytr_extract" # Define here for cleanup scope
        try:
            gdown.download(id=STYTR_CODE_GDRIVE_ID, output=STYTR_ZIP_PATH, quiet=False)
            if not os.path.exists(STYTR_ZIP_PATH):
                 raise FileNotFoundError(f"gdown failed to download {STYTR_ZIP_PATH}")

            progress_bar.progress(20, text="Unzipping StyTR code...")
            if os.path.exists(temp_extract_dir):
                shutil.rmtree(temp_extract_dir)

            with zipfile.ZipFile(STYTR_ZIP_PATH, 'r') as zip_ref:
                zip_ref.extractall(temp_extract_dir)

            extracted_folder_name = find_extracted_code_folder(temp_extract_dir)
            if not extracted_folder_name:
                raise NotADirectoryError(f"Could not find expected StyTR code folder within {temp_extract_dir}. Check zip structure.")

            source_folder_path = os.path.join(temp_extract_dir, extracted_folder_name)

            if os.path.exists(STYTR_DIR):
                st.warning(f"Removing existing '{STYTR_DIR}' directory before replacing.")
                shutil.rmtree(STYTR_DIR)

            shutil.move(source_folder_path, STYTR_DIR)
            shutil.rmtree(temp_extract_dir)
            cleanup_zip()

            progress_bar.progress(40, text="Creating experiments directory...")
            os.makedirs(STYTR_EXPERIMENTS_DIR, exist_ok=True)
            progress_bar.empty()
            st.info("StyTR code downloaded and extracted.")
            return True

        except Exception as e:
            st.error(f"Error during StyTR code download/unzip: {e}")
            cleanup_zip()
            if os.path.exists(temp_extract_dir):
                 shutil.rmtree(temp_extract_dir)
            if 'progress_bar' in locals(): progress_bar.empty()
            return False
    else:
        os.makedirs(STYTR_EXPERIMENTS_DIR, exist_ok=True)
        st.info("StyTR code directory already exists.")
        return True

def setup_stytr_weights():
    """Handles downloading the StyTR weights. Returns True on success."""
    # Corrected paths based on updated IDs
    weights_to_check = {
        "VGG": (STYTR_VGG_WEIGHTS_ID, STYTR_VGG_WEIGHTS_PATH),             # vgg_normalised.pth
        "Embedding": (STYTR_EMBEDDING_WEIGHTS_ID, STYTR_EMBEDDING_WEIGHTS_PATH), # embedding_iter_160000.pth
        "Transformer": (STYTR_TRANSFORMER_WEIGHTS_ID, STYTR_TRANSFORMER_WEIGHTS_PATH), # transformer_iter_160000.pth
    }
    missing_weights = {name: path for name, (_, path) in weights_to_check.items() if not os.path.exists(path)}

    if not missing_weights:
        st.info("All StyTR weights found.")
        return True

    st.info(f"Missing StyTR weights: {', '.join(missing_weights.keys())}. Downloading...")
    progress_bar = st.progress(0, text="Downloading weights...")
    total_weights = len(missing_weights)
    downloaded_count = 0

    for name, (gdown_id, file_path) in weights_to_check.items():
        if name in missing_weights:
            progress_bar.progress(int(downloaded_count / total_weights * 100), text=f"Downloading {name} weights...")
            try:
                if os.path.exists(file_path): os.remove(file_path)
                gdown.download(id=gdown_id, output=file_path, quiet=False)
                if not os.path.exists(file_path):
                     raise FileNotFoundError(f"gdown failed to download {name} weights to {file_path}")
                downloaded_count += 1
            except Exception as e:
                st.error(f"Error downloading {name} weights (ID: {gdown_id}): {e}")
                if os.path.exists(file_path): os.remove(file_path)
                progress_bar.empty()
                return False

    progress_bar.progress(100, text="StyTR weights download complete.")
    time.sleep(1)
    progress_bar.empty()
    return True

def find_extracted_code_folder(base_dir):
    """Finds the main StyTR code folder within the extracted directory."""
    for item in os.listdir(base_dir):
        item_path = os.path.join(base_dir, item)
        if os.path.isdir(item_path):
            expected_util_file = os.path.join(item_path, 'util', 'utils.py')
            if os.path.exists(expected_util_file):
                return item
    for item in os.listdir(base_dir):
        item_path = os.path.join(base_dir, item)
        if os.path.isdir(item_path) and item.startswith('StyTR'):
             return item
    return None

def cleanup_zip():
    """Removes the downloaded zip file if it exists."""
    if os.path.exists(STYTR_ZIP_PATH):
        try:
            os.remove(STYTR_ZIP_PATH)
        except OSError as e:
            st.warning(f"Could not remove zip file {STYTR_ZIP_PATH}: {e}")

@st.cache_resource(show_spinner="Loading Transformer Style Transfer Model...")
def load_stytr_model():
    """Loads the pre-trained StyTR model using cached resource."""
    global stytr_model_instance
    if stytr_model_instance is not None:
        return stytr_model_instance

    if not stytr_initialized:
         st.error("StyTR setup did not complete successfully. Cannot load model.")
         st.stop()

    if _StyTR_model_func is None:
         st.error("StyTR network function reference is missing. Import failed during setup.")
         st.stop()

    st.info("Loading StyTR model weights...")
    try:
        # --- Ensure CWD is correct during model init if it loads files internally ---
        # Although the import worked, the __init__ of the network might also load files
        stytr_dir_abs = os.path.abspath(STYTR_DIR)
        with change_cwd(stytr_dir_abs):
            model = _StyTR_model_func() # Call the imported function
            model.to(DEVICE)
            model.eval()
        # --- CWD restored ---

        stytr_model_instance = model
        st.info("StyTR model loaded.")
        return model
    except FileNotFoundError as e:
        st.error(f"StyTR weight file not found during model loading: {e}. "
                 f"Expected locations within '{STYTR_DIR}/experiments'. ")
        st.stop()
    except Exception as e:
        st.error(f"Error loading StyTR model: {e}")
        import traceback
        st.error(traceback.format_exc())
        st.stop()

def run_stytr_inference(content_file_bytes, style_file_bytes):
    """Runs StyTR inference, handling temporary files."""
    if not stytr_initialized or _stytr_process_images is None:
        st.error("StyTR is not initialized properly. Cannot run inference.")
        return None

    model = load_stytr_model()
    if model is None:
        return None

    temp_content_path = "temp_content_stytr.jpg"
    temp_style_path = "temp_style_stytr.jpg"
    output_img = None

    try:
        with open(temp_content_path, "wb") as f: f.write(content_file_bytes)
        with open(temp_style_path, "wb") as f: f.write(style_file_bytes)

        st.info("Processing images for StyTR...")
        # --- Ensure CWD is correct for process_images if it loads files ---
        stytr_dir_abs = os.path.abspath(STYTR_DIR)
        with change_cwd(stytr_dir_abs):
            content_tensor_stytr, style_tensor_stytr = _stytr_process_images(
                # Pass paths relative to the *new* CWD (StyTR dir) or absolute paths
                os.path.abspath(temp_content_path),
                os.path.abspath(temp_style_path),
                device=DEVICE,
                img_size=IMSIZE
            )
        # --- CWD restored ---

        st.info("Running StyTR model inference...")
        with torch.no_grad():
            # Model inference itself usually doesn't depend on CWD
            output_stytr = model(content_tensor_stytr, style_tensor_stytr)

        if isinstance(output_stytr, (list, tuple)):
            output_tensor = output_stytr[0].cpu()
        else:
            output_tensor = output_stytr.cpu()

        output_img = tensor_to_pil(output_tensor)

    except Exception as e:
        st.error(f"Error during Transformer Style Transfer inference: {e}")
        import traceback
        st.error(traceback.format_exc())
        output_img = None
    finally:
        if os.path.exists(temp_content_path): os.remove(temp_content_path)
        if os.path.exists(temp_style_path): os.remove(temp_style_path)

    return output_img