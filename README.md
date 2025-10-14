# PEFT Repo Metrics

Monitor aggregated metrics of the [PEFT repository](https://github.com/huggingface/peft) and write the data to a [csv stored on Hugging Face Hub](https://huggingface.co/spaces/BenjaminB/peft-repo-metrics/blob/main/metrics.csv). Visualization here: https://huggingface.co/spaces/BenjaminB/peft-repo-metrics.

The code create the fine-grained metrics can be found [here](https://github.com/BenjaminBossan/code-checker). It is required by `analyze.py` to create the aggregate metrics.

Use `backfill.py` to create a monthly interval snapshot of the repo. The included [workflow](https://github.com/BenjaminBossan/peft-repo-metrics/blob/main/.github/workflows/update-metrics.yml) runs nightly to keep the data up to date.
