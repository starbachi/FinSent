# Models Directory

This directory stores trained models and model artifacts.

## Structure
- `checkpoints/` - Training checkpoints (gitignored)
- `final_models/` - Production-ready models (gitignored)
- `configs/` - Model configuration files (tracked)

## Model Storage
All trained models are automatically excluded from git due to size.
Use MLflow or DVC for proper model versioning and storage.

## Loading Models
Models can be loaded using the utilities in `src/models/`
