[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/v3c0XywZ)
# Hand Gesture Recognition System
ECE 4332 / ECE 6332 — AI Hardware  
Fall 2025

## 🧭 Overview
This repository provides a structured template for your team project in the AI Hardware class.  
Each team will **clone this template** to start their own project repository.

## 🗂 Folder Structure
- `ProjectDemo/` – video demonstration of project
- `docs/` – project proposal and documentation  
- `presentations/` – midterm and final presentation slides  
- `src/` – source code for software, hardware, and experiments  
- `data/` – datasets or pointers to data used

## 🧑‍🤝‍🧑 Team Setup
| Name | Role | Responsibilities |
|------|------|------------------|
| Miles Mayhew | Hardware and System integration | Embedded hardware setup, documentation |
| Zackary Dickens | Software | Model optimization |

## 📋 Required Hardware
1. Raspberry Pi 5 and associated power cable
2. USB camera
3. Hailo AI Module
4. Monitor and micro-HDMI to HDMI cable

## 🚀 How to Deploy on Hardware
1. Begin by cloning this repository onto your own personal laptop/device
2. Transfer the good_hand_gesture.tflite or other tflite model of your choosing and piScript.py in the hardware folder to the Raspberry Pi (through scp or thumb stick)
3. In order to run this project python version 3.11.8 must be used. To do this pyenv was used (found here https://github.com/pyenv/pyenv) follow setup steps available on the git to set up on the Raspberry Pi
4. After running pyenv local 3.11.8 in whatever folder you wish to run this project from set up a virtual envirionment and install dependencies using pip install -r requirements.txt
5. Ensure USB camera/Hailo Hat are connected correctly and that the MODEL_PATH variable in piScript.py is correctly pointing towards your chosen model then run with python piScript.py

## 🧾 How to Create a Model
1. Begin by cloning this repository and creating a virtual envirionment then installing dependencies from the requirments.txt file in the model folder (note the model should not be trained on the Raspberry Pi)
2. Can then run train.py which will use the test, train, and validation data from the data folder. If you wish to collect more data the DataCollection script can be deployed on a Raspberry Pi
3. After train.py has completed running a classification report with information about how accurate it is will be printed to the terminal
4. This model can then be deployed using the above steps

## 📜 License
This project is released under the MIT License.
