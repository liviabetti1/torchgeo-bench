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

### Temporal SatCLIP

Edit the relevant model config:

```
src/torchgeo_bench/conf/model/t_satclip/{model}.yaml
```

Update:

- `repo-path` to point to your local satclip repository.
- The checkpoint path to point to the desired temporal SatCLIP checkpoint. (This should be the same for all of us at some point, but right now I'm just playing around with a test checkpoint.)

### GTLoc

From [https://github.com/dshatwell23/gtloc/tree/main](https://github.com/dshatwell23/gtloc/tree/main):

```bash
pip install gdown
gdown --id 170wGyCYLSF3DWOvITIAbtN1OkmDlS4v6 -O /path/to/ckpts/gtloc.pt
```

If needed, edit `conf/models/gtloc.yaml`. It currently assumes the checkpoint is at:

```
/projects/bgtj/t_satclip/ckpts/gtloc/gtloc.pt
```

## 5. Run CoordBench

Return to the torchgeo-bench repository with the environment activated, then run:

```bash
bash src/torchgeo_bench/coordbench/scripts/<your_script>.sh
```