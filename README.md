# Adapted Open Grounding DINO

This repository is a fork of Open GroundingDINO adapted to support fine-tuning with referring expressions
and detailed object descriptions, with the RichArt dataset as a primary target. The goal of this fork is
to enable research and experiments where object instances are described by natural-language references
or long-form descriptions rather than short category labels.

What this fork adds and improves

- Dataset support: utilities and example conversion scripts to transform RichArt-style annotations into
	the training format expected by GroundingDINO. See the `datasets/` folder for dataset loaders and
	format helpers.
- Fine-tuning flow: scripts and helpers to run single-node and distributed fine-tuning, including
	`train_dist.sh` and `test_dist.sh`, example training/validation files in `data/`, and a notebook that
	demonstrates preparing data and running a fine-tuning job.
- Evaluation tooling: an evaluation pipeline that computes detection and retrieval metrics (mAP and
	related scores) for reference-based tasks, integrated into the training loop and available as a
	standalone script in `tools/`.
- Inference and usability fixes: updates to inference utilities to handle richer textual inputs, along
	with runtime fixes and more robust logging to make experimentation easier and reproducible.
- Native ops and build compatibility: adjustments to the native ms_deform_attn implementation and build
	scripts to address newer CUDA toolchains and reduce friction when compiling custom extensions.
- Examples and reproducibility: example notebooks, sample annotation files, and a streamlined setup
	script that helps reproduce training and evaluation runs locally or on multi-GPU instances.

Why this matters for referring-expression tasks

Grounding tasks that use rich textual descriptions require careful dataset formatting, flexible text
tokenization, and evaluation that accounts for language-conditioned localization. This fork brings
together practical additions — dataset loaders, training/evaluation scripts, and inference improvements —
so you can focus on modeling and experiments rather than engineering scaffolding.

Next steps and suggestions

- To reproduce a fine-tuning run: follow the example notebook `training_example.ipynb` and run
	`setup.sh` to install dependencies, then use `train_dist.sh` to start training.
- If you want, I can expand this README with a step-by-step tutorial, a CHANGELOG summarizing the
	development history, or a `CONTRIBUTING.md` describing how to run experiments on RichArt.

