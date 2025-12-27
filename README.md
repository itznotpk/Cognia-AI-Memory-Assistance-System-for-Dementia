# Cognia - AI Memory Assistance System for Dementia

> All project files are now organized in a single directory for easier management and deployment.

## Project Structure

```
Cognia/
├── README.md                # This file
├── Cognia-README.md         # Detailed project documentation
├── index.html               # Caregiver dashboard (now split: see style.css, main.js)
├── style.css                # Extracted CSS from index.html
├── main.js                  # Extracted JS/React code from index.html
├── Scene_Prediction.py      # Desktop/Windows scene detection + API server
├── main.py                  # Raspberry Pi headless version with voice
├── my_model.pt              # YOLO model for kitchen object detection
├── my_model_spec.pt         # YOLO model for spectacles detection
├── presence.json            # Current location state (auto-generated)
├── last_spec_seen.json      # Last spectacles location (auto-generated)
```

## How to Use

- All code, models, and dashboard files are now in the root directory.
- The dashboard's CSS and JavaScript have been separated into `style.css` and `main.js` for maintainability.
- See `Cognia-README.md` for full documentation, setup, and usage instructions.

## Next Steps

- Push this structure to your new repository as soon as you provide the link.
- For further modularization, consider moving Python scripts to a `/src` folder and dashboard files to `/dashboard` if the project grows.
