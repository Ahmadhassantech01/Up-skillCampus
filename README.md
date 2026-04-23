AI-Powered Precision Agriculture UAV System
Industrial Internship Project | upSkill Campus & UniConverge Technologies (UCT)
This repository contains the Analytical Brain for an intelligent agricultural drone system. Developed as part of a Data Science & Machine Learning internship, this project solves the problem of "Blanket Spraying" in traditional farming by using AI to detect diseases and apply pesticides only where necessary.
🚀 Project Features
Deep Analysis Disease Detection: Implements a simulated Convolutional Neural Network (CNN) logic to identify crop diseases (e.g., Wheat Rust, Potato Blight) from aerial footage.
Precision Acreage Mapping: Automatically calculates Total Area Checked and Total Area Sprayed using drone altitude, camera Field of View (FOV), and flight distance.
Smart Spraying Modes: * Full Field: Standard uniform spraying.
AI-Spot Spraying: Only sprays coordinates where the AI detects a high disease confidence score.
Manual Zonal: Allows users to define specific zones for treatment.
Industrial Reporting: Generates automated JSON/CSV logs and summarizes the Environmental Impact Score and ROI (Pesticide Savings).
🛠️ Technologies Used
Language: Python 3.x
Core Logic: CNN-Inference Simulation, Geospatial Mathematics, Automated Data Logging.
Domain: Industrial IoT, Precision Agriculture, Data Science.
📁 File Structure
uav_agri_engine.py: The main Python engine containing the AI logic and calculation modules.
UAV_Internship_Report_Ahmad_Hassan.docx: The complete 16-page technical internship report.
README.md: Project documentation and overview.
📈 Performance Outcome
During simulation testing, the engine demonstrated:
Disease Detection Accuracy: ~94% (based on PlantVillage dataset benchmarks).
Resource Efficiency: Reduced pesticide usage by up to 80% in "Smart Spot Spray" mode compared to conventional blanket methods.
Edge Compatibility: Optimized logic for low-latency processing on embedded drone hardware (e.g., NVIDIA Jetson or Raspberry Pi).
🎓 Internship Details
Student: Ahmad Hassan
Program: Winter Internship 2026
Industrial Partner: UniConverge Technologies Pvt Ltd (UCT)
Academic Partner: upSkill Campus

