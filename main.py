"""
===============================================================
Project :
Facial Expression Recognition Using ConvNeXt or CoAtNet
for Emotion-Driven Music and Mental Health Support

Author : Your Name
===============================================================
"""

import argparse
import torch

from utils.config import Config
from preprocessing.dataset_loader import FERDataLoader

from models.convnext_model import ConvNeXtFER
from models.coatnet import CoAtNetFER

from training.trainer import Trainer
from evaluation.test import Tester

from realtime.webcam import RealTimeFER

from recommendation.music_recommender import MusicRecommender
from mental_health.mood_tracker import MoodTracker


##############################################################
# Select Model
##############################################################

def build_model(model_name):

    if model_name.lower() == "convnext":

        print("\nLoading ConvNeXt Model...\n")

        model = ConvNeXtFER(
            num_classes=Config.NUM_CLASSES
        )

    elif model_name.lower() == "coatnet":

        print("\nLoading CoAtNet Model...\n")

        model = CoAtNetFER(
            num_classes=Config.NUM_CLASSES
        )

    else:
        raise ValueError("Unknown Model")

    return model


##############################################################
# Training
##############################################################

def train(model_name):

    print("="*60)
    print("Training Started")
    print("="*60)

    train_loader, val_loader = FERDataLoader().load()

    model = build_model(model_name)

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader
    )

    trainer.train()

    print("Training Completed")


##############################################################
# Testing
##############################################################

def test(model_name):

    print("="*60)
    print("Testing Started")
    print("="*60)

    test_loader = FERDataLoader().load_test()

    model = build_model(model_name)

    tester = Tester(
        model=model,
        test_loader=test_loader
    )

    tester.evaluate()

    print("Testing Completed")


##############################################################
# Real-Time Webcam FER
##############################################################

def realtime(model_name):

    model = build_model(model_name)

    music = MusicRecommender()

    mood = MoodTracker()

    camera = RealTimeFER(model)

    while True:

        emotion = camera.predict()

        if emotion is None:
            continue

        print("\nDetected Emotion :", emotion)

        ####################################################
        # Music Recommendation
        ####################################################

        song = music.recommend(emotion)

        print("Recommended Music :", song)

        ####################################################
        # Mood Tracking
        ####################################################

        mood.update(emotion)

        mood.display()

        ####################################################
        # Play Music
        ####################################################

        music.play(song)


##############################################################
# Main
##############################################################

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--mode",
        default="train",
        help="train/test/realtime"
    )

    parser.add_argument(
        "--model",
        default="coatnet",
        help="convnext/coatnet"
    )

    args = parser.parse_args()

    print("="*70)
    print("Facial Expression Recognition")
    print("Emotion-Driven Music Recommendation")
    print("Mental Health Support")
    print("="*70)

    print("Selected Model :", args.model)

    if args.mode == "train":

        train(args.model)

    elif args.mode == "test":

        test(args.model)

    elif args.mode == "realtime":

        realtime(args.model)

    else:

        print("Invalid Mode")


##############################################################

if __name__ == "__main__":

    main()
