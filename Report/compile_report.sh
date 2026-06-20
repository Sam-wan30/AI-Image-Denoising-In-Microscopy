#!/bin/bash

# FluoClean AI Report Compilation Script
# This script compiles the LaTeX report and generates a PDF

echo "=========================================="
echo "FluoClean AI Report Compilation"
echo "=========================================="
echo ""

# Check if LaTeX is installed
if ! command -v pdflatex &> /dev/null; then
    echo "Error: pdflatex is not installed."
    echo "Please install TeX Live or MiKTeX first."
    exit 1
fi

# Check if the main file exists
if [ ! -f "FluoClean_AI_Report.tex" ]; then
    echo "Error: FluoClean_AI_Report.tex not found."
    echo "Please run this script from the Report directory."
    exit 1
fi

# Create images directory if it doesn't exist
mkdir -p images

# Create a simple placeholder logo if it doesn't exist
if [ ! -f "placeholder_logo.png" ]; then
    echo "Creating placeholder logo..."
    convert -size 200x200 xc:white -fill black -pointsize 20 -gravity center -annotate +0+0 "LOGO" placeholder_logo.png 2>/dev/null || echo "Note: ImageMagick not found. Using empty placeholder."
fi

echo "Compiling LaTeX document..."
echo ""

# Compile multiple times to resolve references
for i in 1 2 3; do
    echo "Pass $i/3..."
    pdflatex -interaction=nonstopmode FluoClean_AI_Report.tex > compile.log 2>&1

    if [ $? -ne 0 ]; then
        echo "Error during compilation. Check compile.log for details."
        exit 1
    fi
done

echo ""
echo "=========================================="
echo "Compilation successful!"
echo "=========================================="
echo ""
echo "Generated file: FluoClean_AI_Report.pdf"
echo "Total pages: $(pdfinfo FluoClean_AI_Report.pdf 2>/dev/null | grep Pages | awk '{print $2}')"
echo ""

# Clean up auxiliary files
echo "Cleaning up auxiliary files..."
rm -f *.aux *.log *.out *.toc *.lof *.lot *.fls *.fdb_latexmk *.synctex.gz

echo "Done!"
