"""
Energy Efficiency Prediction for Buildings
Bahçeşehir University - Capstone Project

Main entry point for training and evaluating ML models.
Usage: python main.py --data <path_to_dataset.csv>
"""

import argparse
import sys
from pipeline.data_loader import DataLoader
from pipeline.preprocessor import Preprocessor
from pipeline.model_trainer import ModelTrainer
from pipeline.evaluator import Evaluator
from pipeline.predictor import Predictor
from utils.logger import setup_logger

logger = setup_logger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Building Energy Efficiency Prediction System"
    )
    parser.add_argument("--data", type=str, required=True, help="Path to the dataset CSV file")
    parser.add_argument("--mode", type=str, default="train", choices=["train", "predict", "evaluate"],
                        help="Operation mode")
    parser.add_argument("--model", type=str, default="all",
                        choices=["all", "linear", "ridge", "lasso", "elasticnet",
                                 "decision_tree", "random_forest", "gradient_boosting",
                                 "svr", "knn", "mlp"],
                        help="Model to train/use")
    parser.add_argument("--target", type=str, default="both",
                        choices=["heating", "cooling", "both"],
                        help="Target variable to predict")
    parser.add_argument("--output", type=str, default="results/", help="Output directory")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test split ratio")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("Building Energy Efficiency Prediction System")
    logger.info("=" * 60)

    # Step 1: Load data
    logger.info("Step 1: Loading data...")
    loader = DataLoader(args.data)
    df = loader.load()
    loader.summary()

    # Step 2: Preprocess
    logger.info("Step 2: Preprocessing data...")
    preprocessor = Preprocessor(test_size=args.test_size, random_state=args.seed)
    datasets = preprocessor.fit_transform(df, target=args.target)

    # Step 3: Train models
    if args.mode in ("train", "evaluate"):
        logger.info("Step 3: Training models...")
        trainer = ModelTrainer(model_selection=args.model, random_state=args.seed)
        trained_models = trainer.train(datasets)

        # Step 4: Evaluate
        logger.info("Step 4: Evaluating models...")
        evaluator = Evaluator(output_dir=args.output)
        results = evaluator.evaluate(trained_models, datasets)
        evaluator.print_summary(results)
        evaluator.save_results(results)
        evaluator.plot_results(results)

        logger.info(f"Results saved to {args.output}")

    elif args.mode == "predict":
        logger.info("Prediction mode - loading saved model...")
        predictor = Predictor(model_dir=args.output)
        predictor.predict_interactive()

    logger.info("Done!")


if __name__ == "__main__":
    main()
