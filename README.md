# CoD: Black Ops 6 Gameplay Analyzer

This project uses OpenCV and advanced computer vision techniques to analyze Call of Duty: Black Ops 6 gameplay footage. It's designed to detect and track players, weapons, and other in-game elements, providing insights into gameplay strategies and performance.

## Features

* **Object Detection:**  Detects players, weapons, and other key objects in the game using a fine-tuned YOLOv7 model.
* **Object Tracking:**  Tracks detected objects across frames using StrongSORT for robust and accurate tracking.
* **Action Recognition:**  Identifies specific player actions (e.g., reloading, aiming down sights) to analyze gameplay tactics.
* **Pose Estimation:**  Tracks player poses using MediaPipe to understand movement patterns.
* **Minimap Analysis:**  Extracts information from the minimap, such as player positions and objective locations.
* **Experimental Features:** Includes innovative visualizations like the "ghosting" effect to track player movement history.
* **Optimized Performance:**  Leverages OpenCV's optimized functions and Intel Arc GPU acceleration (if available) for smooth and efficient processing.

## Installation


1. **Clone the repository:**
git clone https://github.com/Ac1dBomb/cod_bo6_analyzer

2. Create a conda environment:
conda create -n cod_analyzer python=3.9
conda activate cod_analyzer

3. Install dependencies:
pip install -r requirements.txt

4. Usage
Place your CoD: BO6 gameplay video (cod_bo6_gameplay.mp4) in the project directory.

Run the script:

Bash
python cod_bo6_analyzer.py
Use code with caution.

Press 'q' to exit the analysis.

Customization
You can customize various aspects of the analysis by modifying the parameters in the cod_bo6_analyzer.py script. Refer to the comments within the code for details on available options.

Contributing
Contributions are welcome! Feel free to open issues or submit pull requests to improve the code, add new features, or fix bugs.

License
This project is licensed under the MIT License.


**Remember to replace the following:**

* **`your-username`**:  With your actual GitHub username.
* **`cod_bo6_gameplay.mp4`**: With the actual filename of your gameplay video.

You can further enhance the README by adding:

* **Screenshots or GIFs:** Showcasing the analysis in action.
* **Detailed explanations:**  Providing more context about the algorithms and techniques used.
* **Troubleshooting tips:**  Helping users resolve potential issues.

I'm confident that with your proactive approach, the README will be avaluable guide for anyone interested in using or contributing to our project.
