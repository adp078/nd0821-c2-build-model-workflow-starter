#!/usr/bin/env python
"""
Performs basic cleaning on the data and save the results in Weights & Biases
"""
import argparse
import logging
import wandb
import pandas as pd


logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def go(args):

    run = wandb.init(job_type="basic_cleaning")
    run.config.update(args)

    # Download input artifact. This will also log that this script is using this
    # particular version of the artifact
    # artifact_local_path = run.use_artifact(args.input_artifact).file()

    artifact_local_path = run.use_artifact(args.input_artifact).file()

    dataframe = pd.read_csv(artifact_local_path, index_col="id")
    min_price = args.min_price
    max_price = args.max_price
    idx = dataframe['price'].between(min_price, max_price)
    dataframe = dataframe[idx].copy()
    logger.info("Dataset price outliers removal outside range: %s-%s",
                 args.min_price, args.max_price)
    dataframe['last_review'] = pd.to_datetime(dataframe['last_review'])
    logger.info("Dataset last_review data type fix")

    idx = dataframe['longitude'].between(-74.25, -73.50) & dataframe['latitude'].between(40.5, 41.2)
    dataframe = dataframe[idx].copy()

    filename = "clean_sample.csv"
    dataframe.to_csv(filename, index=False)
    logger.info("Clean artifact saved to %s" , filename)

    artifact = wandb.Artifact(
        args.output_artifact,
        type=args.output_type,
        description=args.output_description
    )

    artifact.add_file(filename)
    run.log_artifact(artifact)

    artifact.wait()
    logger.info("Cleaned dataset uploaded.")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="This steps cleans the data")

    parser.add_argument(
        "--input_artifact", 
        type=str,
        help="Input artifact name",
        required=True
    )

    parser.add_argument(
        "--output_artifact", 
        type=str,
        help="Output artifact name",
        required=True
    )

    parser.add_argument(
        "--output_type", 
        type=str,
        help="Output artifact type",
        required=True
    )

    parser.add_argument(
        "--output_description", 
        type=str,
        help="Output artifact description",
        required=True
    )

    parser.add_argument(
        "--min_price", 
        type=float,
        help="Minimum price limit",
        required=True
    )

    parser.add_argument(
        "--max_price", 
        type=float,
        help="Maximum price limit",
        required=True
    )


    args = parser.parse_args()

    go(args)
