# Neural Style Transfer Showcase

This project demonstrates various neural style transfer techniques using a Streamlit web application interface. It allows users to upload content and style images and generate stylized results using three different methods:

1.  **VGG-based Dual Style Transfer:** Combines features from *two* style images onto a content image.
2.  **VGG-based Rotated Style Transfer:** Applies a modified style (based on feature rotation) from a single style image onto a content image.
3.  **Transformer-based Style Transfer (StyTR):** Utilizes a pre-trained StyTR model for fast, feed-forward style transfer using a single style image.

## Features

*   **Interactive Web UI:** Built with Streamlit for easy image uploads and parameter adjustments.
*   **Multiple Style Transfer Methods:** Implements and showcases three distinct approaches to style transfer.
*   **VGG Optimization Control:** Allows adjusting the number of optimization steps for VGG-based methods.
*   **Side-by-Side Comparison:** Displays original content, style(s), and generated images.
*   **Automatic Setup:** Downloads the required StyTR code and pre-trained model weights automatically on the first run using `gdown`.
*   **Modular Code Structure:** Organizes code into logical modules (`utils`, `vgg_helpers`, `vgg_styles`, `stytr_helpers`).
*   **Device Agnostic:** Runs on CUDA (GPU) if available, otherwise falls back to CPU (GPU highly recommended for VGG methods).

## Demo


## Setup and Installation

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/OxyzGiaHuy/Style_Transfer.git
    cd Style_Transfer
    ```

2.  **Create a Virtual Environment (Recommended):**
    ```bash
    python -m venv venv
    # Activate the environment
    # Windows:
    venv\Scripts\activate
    # macOS/Linux:
    source venv/bin/activate
    ```

3.  **Install Dependencies:**
    Install the required Python packages using the provided `requirements.txt` file:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1.  **Navigate to the Project Directory:**
    Make sure you are in the root directory where `app.py` is located.

2.  **Run the Streamlit Application:**
    ```bash
    streamlit run app.py
    ```

3.  **First Run Setup:**
    *   The first time you run the application, it will automatically download the StyTR codebase and necessary pre-trained weights (`vgg_normalised.pth`, `embedding_iter_160000.pth`, `transformer_iter_160000.pth`) using `gdown`. This may take a few minutes depending on your internet connection.
    *   Wait for the setup process to complete. You should see status messages in the Streamlit interface or console.

4.  **Use the Application:**
    *   The Streamlit interface will open in your web browser.
    *   Use the sidebar to upload:
        *   A **Content Image**.
        *   At least one **Style Image (Style Image 1)**.
        *   Optionally, a second **Style Image (Style Image 2)** to enable the Dual Style method.
    *   Adjust the "Number of VGG Steps" slider if desired (affects Dual and Rotated styles). Higher values improve quality but increase processing time.
    *   Click the "✨ Generate Styled Images" button.
    *   The application will process the images using all applicable methods and display the results.

## File Structure

```
.
├── app.py              # Main Streamlit application file
├── requirements.txt    # Project dependencies
├── README.md           # This file
└── src/                # Source code modules
    ├── __init__.py
    ├── utils.py        # Image loading, tensor conversion utilities
    ├── vgg_helpers.py  # VGG model loading, feature extraction, loss functions
    ├── vgg_styles.py   # Logic for Dual and Rotated VGG styles, optimization loop
    └── stytr_helpers.py # StyTR setup (downloading), model loading, inference logic
```

## Dependencies

All required Python packages are listed in `requirements.txt`. Key dependencies include:

*   `streamlit`
*   `torch` & `torchvision`
*   `Pillow`
*   `gdown`

## Models and Weights

*   **VGG19:** Uses the pre-trained VGG19 model from `torchvision.models` for feature extraction in the VGG-based methods.
*   **StyTR:**
    *   The StyTR codebase is downloaded automatically from a Google Drive link specified in `src/stytr_helpers.py`.
    *   The following pre-trained weights required by StyTR are also downloaded automatically from Google Drive:
        *   `vgg_normalised.pth`
        *   `embedding_iter_160000.pth`
        *   `transformer_iter_160000.pth`

## Notes

*   **GPU Recommended:** While the code runs on CPU, using a CUDA-enabled GPU significantly speeds up the VGG-based style transfer methods.
*   **First Run Time:** The initial setup involving downloads can take several minutes. Subsequent runs will be faster as the necessary files will already exist.
*   **Internet Connection:** An active internet connection is required for the first run to download the StyTR code and weights.
*   **Google Drive IDs:** The application relies on specific Google Drive file IDs defined in `src/stytr_helpers.py`. Ensure these IDs are correct and the linked files are accessible ("Anyone with the link").
