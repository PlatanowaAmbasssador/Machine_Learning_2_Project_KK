# Output Logging and PDF Conversion Guide

## Overview

The project now includes automatic terminal output logging and PDF conversion capabilities. All terminal output from model execution is automatically captured to log files, which can be converted to PDF for submission.

## Features

1. **Automatic Logging**: All stdout/stderr output is automatically captured to log files
2. **Checkpoint Integration**: Logs are automatically appended when resuming from checkpoints
3. **PDF Conversion**: Convert log files to PDF format for submission

## Log File Locations

Log files are automatically saved in:
```
logs/
├── regression/
│   ├── lstm/
│   │   └── lstm_output.log
│   ├── cnn/
│   │   └── cnn_output.log
│   └── ann/
│       └── ann_output.log
└── classification/
    ├── lstm/
    │   └── lstm_output.log
    ├── cnn/
    │   └── cnn_output.log
    └── ann/
        └── ann_output.log
```

## Usage

### Running Models (Automatic Logging)

When you run any executor script, logging is automatically enabled:

```bash
# Regression models
python executors/regression/run_lstm.py
python executors/regression/run_cnn.py
python executors/regression/run_ann.py

# Classification models
python executors/classification/run_lstm.py
python executors/classification/run_cnn.py
python executors/classification/run_ann.py
```

All output will be:
- Displayed in the terminal (as normal)
- Saved to the log file automatically
- Appended if resuming from a checkpoint

### Converting Logs to PDF

After execution completes, convert the log file to PDF:

```bash
# Basic usage
python utils/log_to_pdf.py logs/regression/lstm/lstm_output.log

# Specify output file
python utils/log_to_pdf.py logs/regression/lstm/lstm_output.log -o output.pdf

# Custom title
python utils/log_to_pdf.py logs/regression/lstm/lstm_output.log -t "LSTM Model Execution Log"

# Text-only output (if reportlab not installed)
python utils/log_to_pdf.py logs/regression/lstm/lstm_output.log --text-only
```

### Checkpoint Resume and Log Combining

When you resume execution from a checkpoint:

1. The system automatically detects if a log file exists
2. New output is **appended** to the existing log file
3. A separator is added to distinguish between runs
4. The combined log can be converted to PDF

Example:
```bash
# First run (interrupted after fold 2)
python executors/regression/run_lstm.py
# ... execution interrupted ...

# Resume (logs will be appended)
python executors/regression/run_lstm.py
# ... execution completes ...

# Convert combined log to PDF
python utils/log_to_pdf.py logs/regression/lstm/lstm_output.log
```

## Requirements

### For PDF Conversion

The PDF conversion uses `reportlab`. Install it with:

```bash
pip install reportlab
```

If `reportlab` is not installed, the script will create a formatted text file instead, which you can manually convert to PDF.

### Manual PDF Conversion (Alternative)

If you prefer not to install `reportlab`, you can:

1. Use the `--text-only` flag to create a formatted text file
2. Convert manually:
   - **macOS**: Open in TextEdit and Print to PDF
   - **Linux**: Use `enscript` or `a2ps`: `enscript -p output.ps file.txt && ps2pdf output.ps`
   - **Windows**: Print to PDF from any text editor

## Log File Format

Log files include:
- All print statements
- TensorFlow/Keras training progress
- Model selection information
- Fold completion summaries
- Final results and metrics
- Timestamps for resume points

## Tips

1. **Large Logs**: Very large log files (>100MB) may take time to convert to PDF. Consider using `--text-only` for very large files.

2. **Multiple Models**: Each model has its own log file, so you can convert them separately or combine them manually if needed.

3. **Checkpoint Safety**: Logs are saved after each fold, so even if execution is interrupted, you won't lose output.

4. **Log Cleanup**: Old log files can be safely deleted if you want to start fresh. The system will create new ones automatically.

## Troubleshooting

### Log file not created
- Check that the `logs/` directory is writable
- Ensure the executor script ran successfully

### PDF conversion fails
- Install `reportlab`: `pip install reportlab`
- Use `--text-only` flag as fallback
- Check that the log file exists and is readable

### Logs not combining on resume
- Ensure `resume=True` in the executor script
- Check that checkpoint directory exists
- Verify log file path matches checkpoint directory structure

