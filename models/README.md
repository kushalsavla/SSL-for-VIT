# Models Directory

This directory contains model-related files and configurations.

## Vision Transformer Location

**Important**: The vision transformer implementation is now located in the external DINO v2 repository to avoid duplication and ensure we use the official implementation.

### Path: `external/dino/dinov2/models/vision_transformer.py`

### Navigation Structure:
```
external/
└── dino/                    # DINO v2 repository
    └── dinov2/              # Main DINO v2 package
        └── models/          # Model implementations
            └── vision_transformer.py  # Vision Transformer implementation
```

### Import Usage:
```python
# Add the dinov2 path to sys.path
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'external', 'dino', 'dinov2'))

# Import the vision transformer
from models.vision_transformer import vit_small, vit_base, vit_large
```

### Available Models:
- `vit_small`: Small Vision Transformer (384 dim, 12 layers, 6 heads)
- `vit_base`: Base Vision Transformer (768 dim, 12 layers, 12 heads)
- `vit_large`: Large Vision Transformer (1024 dim, 24 layers, 16 heads)

### Why External?
- **No Duplication**: Avoids maintaining duplicate vision transformer code
- **Official Implementation**: Uses the official DINO v2 implementation
- **Updates**: Automatically gets updates from the official repository
- **Consistency**: Ensures all SSL methods use the same base architecture

### Setup:
The external DINO v2 repository is automatically cloned by the `setup.sh` script:
```bash
./setup.sh
```

This will clone the DINO v2 repository to `external/dino/` and set up all necessary dependencies. 