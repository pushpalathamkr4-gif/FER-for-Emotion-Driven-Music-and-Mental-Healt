"""
===============================================================
Project :
Facial Expression Recognition Using ConvNeXt or CoAtNet
for Emotion-Driven Music and Mental Health Support

File : config.py

Author : Your Name
===============================================================
"""

import os
import torch


class Config:

    ###############################################################
    # Project Information
    ###############################################################

    PROJECT_NAME = "FER_Emotion_Music_MentalHealth"

    VERSION = "1.0"

    AUTHOR = "Your Name"

    ###############################################################
    # Device Configuration
    ###############################################################

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    NUM_WORKERS = 4

    PIN_MEMORY = True

    ###############################################################
    # Dataset Paths
    ###############################################################

    ROOT_DIR = os.getcwd()

    DATASET_DIR = os.path.join(ROOT_DIR, "datasets")

    AFFECTNET_DIR = os.path.join(DATASET_DIR, "AffectNet")

    RAFDB_DIR = os.path.join(DATASET_DIR, "RAF_DB")

    FER2013_DIR = os.path.join(DATASET_DIR, "FER2013")

    ###############################################################
    # Output Directories
    ###############################################################

    OUTPUT_DIR = os.path.join(ROOT_DIR, "results")

    MODEL_DIR = os.path.join(OUTPUT_DIR, "saved_models")

    LOG_DIR = os.path.join(OUTPUT_DIR, "logs")

    GRAPH_DIR = os.path.join(OUTPUT_DIR, "graphs")

    CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, "checkpoints")

    ###############################################################
    # Image Parameters
    ###############################################################

    IMAGE_SIZE = 224

    CHANNELS = 3

    ###############################################################
    # Emotion Classes
    ###############################################################

    EMOTIONS = [

        "Happy",
        "Sad",
        "Angry",
        "Fear",
        "Neutral",
        "Surprise",
        "Disgust"

    ]

    NUM_CLASSES = len(EMOTIONS)

    ###############################################################
    # Training Parameters
    ###############################################################

    BATCH_SIZE = 32

    EPOCHS = 50

    LEARNING_RATE = 1e-4

    WEIGHT_DECAY = 1e-4

    MOMENTUM = 0.9

    ###############################################################
    # Optimizer
    ###############################################################

    OPTIMIZER = "AdamW"

    LOSS_FUNCTION = "CrossEntropyLoss"

    ###############################################################
    # Scheduler
    ###############################################################

    LR_SCHEDULER = "CosineAnnealingLR"

    ###############################################################
    # ConvNeXt Parameters
    ###############################################################

    CONVNEXT_MODEL = "convnext_tiny"

    ###############################################################
    # CoAtNet Parameters
    ###############################################################

    COATNET_MODEL = "coatnet0"

    ###############################################################
    # Early Stopping
    ###############################################################

    EARLY_STOPPING = True

    PATIENCE = 10

    ###############################################################
    # Random Seed
    ###############################################################

    SEED = 42

    ###############################################################
    # Webcam
    ###############################################################

    CAMERA_ID = 0

    FRAME_WIDTH = 640

    FRAME_HEIGHT = 480

    ###############################################################
    # Music Recommendation
    ###############################################################

    MUSIC_LIBRARY = {

        "Happy": "music/happy/",
        "Sad": "music/sad/",
        "Angry": "music/relax/",
        "Fear": "music/meditation/",
        "Neutral": "music/neutral/",
        "Surprise": "music/energetic/",
        "Disgust": "music/calm/"

    }

    ###############################################################
    # Mental Health Alerts
    ###############################################################

    STRESS_THRESHOLD = 3

    SADNESS_THRESHOLD = 5

    ###############################################################
    # Logging
    ###############################################################

    SAVE_MODEL = True

    SAVE_CHECKPOINT = True

    SAVE_GRAPHS = True

    SAVE_LOGS = True

    ###############################################################
    # Create Required Directories
    ###############################################################

    @staticmethod
    def create_directories():

        directories = [

            Config.OUTPUT_DIR,
            Config.MODEL_DIR,
            Config.LOG_DIR,
            Config.GRAPH_DIR,
            Config.CHECKPOINT_DIR

        ]

        for directory in directories:

            os.makedirs(directory, exist_ok=True)

    ###############################################################
    # Display Configuration
    ###############################################################

    @staticmethod
    def show():

        print("=" * 60)
        print("PROJECT CONFIGURATION")
        print("=" * 60)

        print(f"Project Name   : {Config.PROJECT_NAME}")
        print(f"Version        : {Config.VERSION}")
        print(f"Device         : {Config.DEVICE}")
        print(f"Image Size     : {Config.IMAGE_SIZE}")
        print(f"Batch Size     : {Config.BATCH_SIZE}")
        print(f"Epochs         : {Config.EPOCHS}")
        print(f"Learning Rate  : {Config.LEARNING_RATE}")
        print(f"Optimizer      : {Config.OPTIMIZER}")
        print(f"Loss Function  : {Config.LOSS_FUNCTION}")
        print(f"Number Classes : {Config.NUM_CLASSES}")

        print("=" * 60)


###############################################################
# Main
###############################################################

if __name__ == "__main__":

    Config.create_directories()

    Config.show()
