# University of Virginia
## Department of Electrical and Computer Engineering

**Course:** ECE 4332 / ECE 6332 — AI Hardware Design and Implementation  
**Semester:** Fall 2025  
**Proposal Deadline:** November 5, 2025 — 11:59 PM  
**Submission:** Upload to Canvas (PDF) and to GitHub (`/docs` folder)

---

# AI Hardware Project Proposal Template

## 1. Project Title
ECE4380Group project

Miles Mayhew, Zackary Dickens

Gesture Recognition System

## 2. Platform Selection
We selected the Raspberry Pi AI Kit since it has a good balance of performance, flexibility, and usability. This platform can perform low-latency inference on visual data and the Raspberry Pi supports a wide range of cameras which can be used to collect data.

**Undergraduates:** Edge-AI, TinyML, or Neuromorphic platforms  
**Graduates:** open-source AI accelerators (Ztachip, VTA, Gemmini, VeriGOOD-ML, NVDLA) or any of the above 

## 3. Problem Definition
The problem which we are addressing is that traditional input methods (like mouse and keyboard) can be limiting for real-time hands-free applications. Our project will address this by recognizing hand gestures and translating them into computer commands using edge-AI. This project will demonstrate how AI hardware can enable low-latency, scalable, and interactive embedded AI solutions.

## 4. Technical Objectives
1. We should be able to identify the thumbs up, open palm, fist, and pointing gestures. 
2. We should be able to identify the gestures more than 90% of the time.
3. The Raspberry Pi should print each gesture after detecting it within 150 ms. 
4. Model should be optimized to run entirely on-device.

## 5. Methodology
The hardware setup will include the Raspberry Pi 5 with the AI kit featuring the Hailo-8L NPU for accelerated AI inference. A Raspberry Pi camera module will provide real time data while the Raspberry pi is connected to a display through micro HDMI for visual output. The setup will operate entirely on the device without cloud dependency. 
  
The project will be developed in python using libraries like OpenCV for video processing and TensorFlow Lite for model deployment. Additional tools include Matplotlib for visualizing results. A lightweight convolutional neural network (CNN) trained on a hand gesture dataset will serve as the recognition model which will classify gestures like thumbs up, open hand, and closed fist. The model will be compiled for hardware acceleration on the AI Kit.

The performance will be evaluated using the metrics of inference latency (time to output result from an input image) and classification accuracy. The validation strategy will involve testing the system under different environmental conditions like varying lighting (light, dim, and dark) and varying users to assess its robustness. Each gesture will be performed multiple times and the performance results will be evaluated based off accuracy percentage and average inference latency. 

## 6. Expected Deliverables
- Working demo of the system correctly identifying  hand gestures
- GitHub repository with comprehensive readme
- Documentation on the system and how to deploy it
- Presentation Slides
- Final Report

## 7. Team Responsibilities

| Name | Role | Responsibilities |
|------|------|------------------|
| Miles Mayhew | Hardware and System integration | Embedded hardware setup, documentation |
| Zackary Dickens | Software | Model optimization |

## 8. Timeline and Milestones

| Week | Milestone | Deliverable |
|------|------------|-------------|
| 2 | Proposal | PDF + GitHub submission |
| 4 | Midterm presentation | Slides, preliminary results |
| 6 | Integration & testing | Working prototype |
| Dec. 18 | Final presentation | Report, demo, GitHub archive |

## 9. Resources Required
Will need a  Raspberry Pi 5, Hailo-8L NPU, and a camera compatible with the Raspberry Pi 5.

## 10. References
Include relevant papers, repositories, and documentation.
