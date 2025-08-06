# External Dependencies

This directory contains external repositories that need to be cloned separately.

## Required Repositories

Run the following commands to set up external dependencies:

```bash
# Clone iBOT repository
git clone https://github.com/bytedance/ibot.git ibot

# Clone DINO v2 repository  
git clone https://github.com/facebookresearch/dinov2.git dino
```

## Structure After Setup

```
external/
├── ibot/          # iBOT SSL implementation
├── dino/          # DINO v2 SSL implementation (dinov2/)
└── README.md      # This file
```

## Documentation

For detailed setup instructions, see [docs/EXTERNAL_DEPENDENCIES.md](../docs/EXTERNAL_DEPENDENCIES.md) and [docs/SETUP_INSTRUCTIONS.md](../docs/SETUP_INSTRUCTIONS.md). 