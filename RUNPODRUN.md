# Running the pipeline on Runpod

CPU-only. Runtime and data stay on the pod, results come back to `./data` here.

## Requirements

- Runpod account with credit, and your SSH public key added under
  **Settings -> SSH Public Keys** *before* creating the pod (`cat ~/.ssh/id_ed25519.pub`).
- Kaggle API token (`KGAT_...`, kaggle.com -> Settings -> API -> Create New Token).
  Open the [dataset page](https://www.kaggle.com/datasets/ealtman2019/credit-card-transactions)
  once and accept its terms if asked.
- Pod: **CPU**, 4+ vCPU, **16 GB+ RAM**, **20 GB+ disk** (raw CSV is ~2.3 GB).

## 1. Create the pod (Runpod console)

1. **Pods -> Deploy -> CPU**, pick a 4 vCPU / 16 GB (or larger) instance.
2. Template: any Ubuntu image (the default works). Set volume disk to 20 GB at `/workspace`.
3. Deploy, wait for *Running*, then **Connect** and copy the
   **SSH over exposed TCP** command. It looks like `ssh root@<IP> -p <PORT> -i ~/.ssh/id_ed25519`.

Below, replace `<IP>` and `<PORT>` with those values.

## 2. Upload the code (from this folder, on your Mac)

```bash
tar czf - --exclude .venv --exclude data --exclude .env \
  --exclude __pycache__ --exclude .pytest_cache --exclude .git . \
| ssh -p <PORT> -i ~/.ssh/id_ed25519 root@<IP> \
  'mkdir -p /workspace/ow_banking_project && tar xzf - -C /workspace/ow_banking_project'
```

## 3. Run (on the pod)

```bash
ssh -p <PORT> -i ~/.ssh/id_ed25519 root@<IP>

cd /workspace/ow_banking_project
curl -LsSf https://astral.sh/uv/install.sh | sh && source $HOME/.local/bin/env
export KAGGLE_API_TOKEN=KGAT_...            # type it here, don't upload .env

uv sync                                      # installs Python 3.14 + deps
uvx kaggle datasets download ealtman2019/credit-card-transactions -p data/raw --unzip
uv run python -m reco ingest                 # CSV -> data/warehouse (Parquet)
uv run python -m reco train                  # -> data/user_scores.parquet, data/segment.joblib
uv run pytest
uv run --with nbconvert jupyter nbconvert --to notebook --execute --inplace notebooks/report.ipynb
```

Use `tmux` (or `nohup ... &`) for `ingest`/`train` if your connection may drop.

## 4. Download the results (on your Mac, from this folder)

```bash
ssh -p <PORT> -i ~/.ssh/id_ed25519 root@<IP> \
  'cd /workspace/ow_banking_project && tar czf - --exclude data/raw data notebooks/report.ipynb' \
| tar xzf -
```

You get `data/user_scores.parquet`, `data/segment.joblib`, `data/warehouse/` (needed by the
notebook) and the executed `notebooks/report.ipynb`.

## 5. Stop paying

Runpod console -> the pod -> **Terminate**. Stopped pods still bill for disk.
