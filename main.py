#!/usr/bin/env python3
# main.py

import argparse
from pathlib import Path

import pandas as pd

from src.data_loader  import load_data
from src.preprocessing import prepare_data, clean_corpus
from src.sentiment     import analyze_sentiment, label_sentiment
from src.clustering   import pipeline_cluster

def main():
    parser = argparse.ArgumentParser(
        description="NLP Insights Pipeline: load → preprocess → sentiment → cluster"
    )
    parser.add_argument("--path",       type=str,   default="data/raw/Reviews.csv",
                        help="path to Reviews.csv")
    parser.add_argument("--nrows",      type=int,   default=None,
                        help="number of rows to read (for quick tests)")
    parser.add_argument("--mode",       type=str,
                        choices=["load","preprocess","sentiment","cluster","full"],
                        default="full",
                        help="which stage to run")
    parser.add_argument("--sample",     type=int,   default=10000,
                        help="sample size for sentiment & clustering")
    args = parser.parse_args()

    # 1) Load
    df = load_data(args.path, nrows=args.nrows)
    if args.mode == "load":
        print(df.head())
        return

    # 2) Preprocess
    df = prepare_data(df)
    df["cleaned"] = clean_corpus(df["Text"].tolist(), batch_size=500, n_process=2)
    if args.mode == "preprocess":
        print(df[["Text","cleaned"]].head())
        return

    # 3) Sentiment
    df_sample = df.sample(n=args.sample, random_state=42).reset_index(drop=True)
    scores = analyze_sentiment(df_sample["cleaned"].tolist())
    df_sample["vader_compound"] = scores
    df_sample["vader_label"]    = label_sentiment(scores)
    if args.mode == "sentiment":
        print(df_sample[["vader_label"]].value_counts())
        return

    # 4) Clustering
    embeds2d, labels = pipeline_cluster(
        df_sample["cleaned"].tolist(),
        umap_components=2,
        min_cluster_size=15
    )
    df_sample["cluster"] = labels
    if args.mode == "cluster":
        print("Clusters distribution:\n", df_sample["cluster"].value_counts())
        return

    # 5) Full pipeline
    # Save sample with all annotations
    out = Path("output_sample.csv")
    df_sample.to_csv(out, index=False)
    print(f"Full pipeline done — sample saved to {out}")

if __name__ == "__main__":
    main()
