# TorchGeo-Bench Setup

## 1. Clone torchgeo-bench

From your home directory:

```bash
cd ~
git clone git@github.com:liviabetti1/torchgeo-bench.git
cd torchgeo-bench
git checkout temporal
```

## 2. Set up the environment (with uv)

Install uv:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Then, from the torchgeo-bench project root:

```bash
uv sync --extra dev --extra coordbench
```

This creates a `.venv` in the project root. Activate it with:

```bash
source .venv/bin/activate
```

> **Note:** Veevee seems to have another way of doing this, but this is what works for me / I forget how he did what he did...

## 3. Clone satclip

From your home directory:

```bash
cd ~
git clone git@github.com:arizonat/satclip.git
cd satclip
git checkout temporal
```

## 4. Configure models

### Setting up your environment
You'll need to export paths for:
- your virtual environment
- your local torchgeo repository
- your local satclip repository

For example, when running on rails, I do the following:
```bash
export TORCHGEO_HOME=/u/kplane19/torchgeo-bench
export TORCHGEO_ENV=/u/kplane19/torchgeo-bench/.venv
export SATCLIP_HOME=/u/kplane19/satclip
export RESULTS_HOME=/projects/bgtj/t_satclip/results
export CKPT_HOME=/projects/bgtj/t_satclip/ckpts
```

The variable names must be TORCHGEO_HOME, TORCHGEO_ENV, SATCLIP_HOME, RESULTS_HOME, and CKPT_HOME for compatibility with our scripts.  For simplicity's sake, you can put these lines in the bottom of your .bashrc file (found inside your /u/username/ directory).  Upon login, they will automatically be added to your environment.  These can (and should) vary from machine to machine as a way to avoid hard-coding relative paths. 

If you are writing your own sbatch .job script, you MUST have the line #sbatch --export=ALL on in order for the job to pick up your local environment variables.

### GTLoc

From [https://github.com/dshatwell23/gtloc/tree/main](https://github.com/dshatwell23/gtloc/tree/main):

```bash
pip install gdown
gdown --id 170wGyCYLSF3DWOvITIAbtN1OkmDlS4v6 -O /path/to/ckpts/gtloc.pt
```

## 5. Run CoordBench

Return to the torchgeo-bench repository with the environment activated, then run:

```bash
bash src/torchgeo_bench/coordbench/scripts/<your_script>.sh
```