# CoD: Black Ops 6 Gameplay Analyzer

This project utilizes OpenCV and advanced computer vision techniques to analyze Call of Duty: Black Ops 6 gameplay footage. It's designed to detect, track players, weapons, and other in-game elements, providing insights into gameplay strategies and performance. This project leverages Google Cloud Platform (GCP), Google Cloud Storage, Cloud SQL, and potentially other Google Cloud services.

## Features

* **Object Detection:** Detects players, weapons, and other key objects in the game using a fine-tuned YOLOv7 model.
* **Object Tracking:** Tracks detected objects across frames using StrongSORT for robust and accurate tracking.
* **Action Recognition:** Identifies specific player actions (e.g., reloading, aiming down sights) to analyze gameplay tactics.
* **Pose Estimation:** Tracks player poses using MediaPipe to understand movement patterns.
* **Minimap Analysis:** Extracts information from the minimap, such as player positions and objective locations.
* **Experimental Features:** Includes innovative visualizations like the "ghosting" effect to track player movement history.
* **Optimized Performance:** Leverages OpenCV's optimized functions and Intel Arc GPU acceleration (if available) for smooth and efficient processing.
* **Google Cloud Integration:** Utilizes Google Cloud Storage for storing code, data, and models, and Cloud SQL for MySQL as the database solution.
* **Gemini API Integration:** Integrates with the Gemini API to leverage its natural language processing and code generation capabilities.

## Installation

### Google Cloud Platform (GCP) Setup

1. **Set up a Google Cloud Project:**
   * Create a new project in your Google Cloud console.
   * [Link to Google Cloud Console](https://console.cloud.google.com/)

2. **Enable APIs:**
   * Enable the following APIs in your project:
     * Cloud Storage
     * Cloud SQL Admin API
     * Compute Engine API
     * Cloud Vision API
     * Cloud Natural Language API
     * Cloud Video Intelligence API
     * Gemini API (See Google Cloud documentation for instructions)
   * [Link to Google Cloud APIs](https://console.cloud.google.com/apis/library)

3. **Install gcloud CLI:**
   * Download and install the Google Cloud CLI.
   * [Link to gcloud CLI download](https://cloud.google.com/sdk/docs/install)

4. **Authenticate with gcloud:**
   * Run `gcloud auth login` in your terminal to authenticate with your Google Cloud account.

5. **Set your project:**
   * Run `gcloud config set project [your-project-id]` to set the active project for the gcloud CLI.

6. **Create a Cloud Storage Bucket:**
   * Create a bucket with Hierarchical Namespace enabled.
   * [Link to Cloud Storage documentation](https://cloud.google.com/storage/docs)

7. **Create a Cloud SQL Instance:**
   * Create a Cloud SQL instance for MySQL.
   * [Link to Cloud SQL documentation](https://cloud.google.com/sql/docs/mysql)

### Local Setup

8. **Install Git:**
   * **Windows:** Download and install Git for Windows from the official website.
     * [Link to Git for Windows download](https://git-scm.com/download/win)
   * **Linux:** Use your distribution's package manager to install Git (e.g., `sudo apt-get install git` on Debian/Ubuntu).

9. **Clone the Repository:**
   ```bash
   git clone https://github.com/Ac1dBomb/cod_bo6_analyzer

10. **Create a Virtual Environment (Recommended):**

Windows:
Bash
python -m venv .venv
.venv\Scripts\activate
Use code with caution.

Linux:
Bash
python3 -m venv .venv
source .venv/bin/activate
Use code with caution.

Install Dependencies:

Bash
pip install -r requirements.txt
Use code with caution.

11. **Usage:**
Place your CoD: BO6 gameplay video (cod_bo6_gameplay.mp4) in the project directory.
Update the cod_bo6_analyzer.py script:
Replace "yolov7.pt" and "yolov7_quant.tflite" with the actual paths to your YOLOv7 and TensorFlow Lite models.
Replace "path/to/your/video.mp4" with the actual path to your gameplay video.
Implement the placeholder functions (track_objects, analyze_actions, analyze_minimap, analyze_with_gemini, draw_visualizations) with your desired logic.
Run the script:
Bash
python cod_bo6_analyzer.py
Use code with caution.

Press 'q' to exit the analysis.
GitHub
While this project is primarily developed and deployed on Google Cloud, we encourage you to explore our GitHub repository for:

Code Updates: We'll periodically update the GitHub repository with the latest code and documentation.
Community Contributions: Feel free to contribute to the project by opening issues, submitting bug reports, or proposing new features.
Discussions and Collaboration: Engage with the community and share your ideas or feedback on the project.
[Link to GitHub Repository]

Customization
You can customize various aspects of the analysis by modifying the parameters in the cod_bo6_analyzer.py script. Refer to the comments within the code for details on available options.

Contributing
Contributions are welcome! Feel free to open issues or submit pull requests to improve the code, add new features, or fix bugs.

License
This project is licensed under the MIT License.
**Key changes**

*   Added step-by-step instructions for installing Git on both Windows and Linux.
*   Included commands for creating and activating a virtual environment on Windows and Linux.
*   Clarified the process of cloning the repository and installing dependencies.

**Remember to replace the following:**

* **`your-username`**:  With your actual GitHub username.
* **`cod_bo6_gameplay.mp4`**: With the actual filename of your gameplay video.