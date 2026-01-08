#!/usr/bin/env python3
"""
Convert log files to PDF format for submission.
"""

import os
import sys
import argparse
from pathlib import Path
from datetime import datetime

try:
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Preformatted, PageBreak
    from reportlab.lib.enums import TA_LEFT, TA_CENTER
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttf import TTFont
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False
    print("Warning: reportlab not installed. Install with: pip install reportlab")
    print("Falling back to text-based output.")


def log_to_pdf_text(log_file: str, output_file: str):
    """
    Simple text-based conversion (fallback if reportlab not available).
    Creates a formatted text file that can be converted to PDF manually.
    """
    with open(log_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Create formatted text output
    header = f"""
{'='*80}
TERMINAL OUTPUT LOG
{'='*80}
Source File: {log_file}
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
{'='*80}

"""
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(header)
        f.write(content)
    
    print(f"Text file created: {output_file}")
    print("You can convert this to PDF using:")
    print("  - macOS: Open in TextEdit and Print to PDF")
    print("  - Linux: Use 'enscript' or 'a2ps'")
    print("  - Windows: Print to PDF from any text editor")


def log_to_pdf_reportlab(log_file: str, output_file: str, title: str = "Terminal Output"):
    """
    Convert log file to PDF using reportlab.
    
    Parameters:
    -----------
    log_file : str
        Input log file path
    output_file : str
        Output PDF file path
    title : str
        PDF title
    """
    if not REPORTLAB_AVAILABLE:
        log_to_pdf_text(log_file, output_file.replace('.pdf', '.txt'))
        return
    
    # Read log file
    with open(log_file, 'r', encoding='utf-8') as f:
        log_content = f.read()
    
    # Create PDF
    doc = SimpleDocTemplate(
        output_file,
        pagesize=letter,
        rightMargin=72,
        leftMargin=72,
        topMargin=72,
        bottomMargin=18
    )
    
    # Container for the 'Flowable' objects
    elements = []
    
    # Define styles
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=16,
        textColor='#000000',
        spaceAfter=30,
        alignment=TA_CENTER
    )
    
    header_style = ParagraphStyle(
        'CustomHeader',
        parent=styles['Normal'],
        fontSize=10,
        textColor='#666666',
        spaceAfter=20
    )
    
    code_style = ParagraphStyle(
        'CodeStyle',
        parent=styles['Code'],
        fontSize=8,
        leading=10,
        fontName='Courier',
        leftIndent=0,
        rightIndent=0
    )
    
    # Add title
    elements.append(Paragraph(title, title_style))
    elements.append(Spacer(1, 0.2*inch))
    
    # Add metadata
    metadata = f"""
    <b>Source File:</b> {os.path.basename(log_file)}<br/>
    <b>Generated:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}<br/>
    <b>File Size:</b> {os.path.getsize(log_file) / 1024:.2f} KB
    """
    elements.append(Paragraph(metadata, header_style))
    elements.append(Spacer(1, 0.3*inch))
    
    # Split content into lines and add as preformatted text
    # Process in chunks to avoid memory issues with very large files
    lines = log_content.split('\n')
    chunk_size = 1000  # Process 1000 lines at a time
    
    for i in range(0, len(lines), chunk_size):
        chunk = '\n'.join(lines[i:i+chunk_size])
        # Escape HTML special characters for Paragraph
        chunk = chunk.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
        
        # Use Preformatted for code-like output
        pre = Preformatted(chunk, code_style, maxLineLength=100)
        elements.append(pre)
        
        # Add page break between large chunks
        if i + chunk_size < len(lines):
            elements.append(PageBreak())
    
    # Build PDF
    doc.build(elements)
    print(f"PDF created: {output_file}")


def main():
    parser = argparse.ArgumentParser(description='Convert log files to PDF')
    parser.add_argument('log_file', type=str, help='Input log file path')
    parser.add_argument('-o', '--output', type=str, help='Output PDF file path (default: log_file.pdf)')
    parser.add_argument('-t', '--title', type=str, default='Terminal Output', help='PDF title')
    parser.add_argument('--text-only', action='store_true', help='Create text file instead of PDF')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.log_file):
        print(f"Error: Log file not found: {args.log_file}")
        sys.exit(1)
    
    if args.output:
        output_file = args.output
    else:
        output_file = args.log_file.replace('.log', '.pdf')
        if output_file == args.log_file:
            output_file = args.log_file + '.pdf'
    
    if args.text_only:
        log_to_pdf_text(args.log_file, output_file.replace('.pdf', '.txt'))
    else:
        log_to_pdf_reportlab(args.log_file, output_file, args.title)


if __name__ == '__main__':
    main()

