# Quick Start Guide - FluoClean AI Report

This guide will help you quickly compile and customize the FluoClean AI research report.

## Immediate Actions

### 1. Customize Personal Information
Open `FluoClean_AI_Report.tex` and replace ALL placeholders:
- `[Student Name]` → Your full name
- `[Roll Number]` → Your roll number
- `[Guide Name]` → Project guide's name
- `[Designation]` → Guide's designation (e.g., Professor, Associate Professor)
- `[Department]` → Department name (e.g., Department of Computer Science)
- `[Head of Department]` → HOD's name
- `[University/College Name]` → Your institution name
- `[Academic Year 2024-2025]` → Current academic year

### 2. Add Institution Logo (Optional)
Replace `placeholder_logo.png` with your institution's logo:
- Recommended size: 200×200 pixels
- Format: PNG or JPG
- Place in the Report directory
- Name it exactly: `placeholder_logo.png`

Or remove the logo line from the title page if not needed.

### 3. Compile the Report

#### Option A: Using the Script (Recommended)
```bash
cd "/Users/samiksha/AI Image Denoising In Microscopy/Report"
chmod +x compile_report.sh
./compile_report.sh
```

#### Option B: Manual Compilation
```bash
cd "/Users/samiksha/AI Image Denoising In Microscopy/Report"
pdflatex FluoClean_AI_Report.tex
pdflatex FluoClean_AI_Report.tex
pdflatex FluoClean_AI_Report.tex
```

#### Option C: Using Overleaf (Easiest)
1. Go to [overleaf.com](https://www.overleaf.com)
2. Create a new project
3. Upload `FluoClean_AI_Report.tex`
4. Click "Recompile"
5. Download the PDF

## Troubleshooting

### "File not found" errors
- Ensure you're in the Report directory
- Check that `FluoClean_AI_Report.tex` exists

### "Package not found" errors
Install missing LaTeX packages:
```bash
tlmgr install graphicx amsmath geometry fancyhdr titlesec tocloft array booktabs caption subcaption hyperref tikz
```

### Compilation takes too long
- This is normal for the first compilation
- Subsequent compilations are faster
- Total time: 30-60 seconds on modern computers

### PDF has blank pages or missing content
- Compile 3 times (references need multiple passes)
- Check the compile.log file for errors

## Customization Tips

### Changing Page Margins
In the LaTeX document, find:
```latex
\geometry{left=1.5in,right=1in,top=1in,bottom=1in}
```
Adjust values as needed.

### Adding Your Own Images
1. Create an `images/` subdirectory
2. Add your image files
3. In the .tex file, replace TikZ diagrams with:
```latex
\begin{figure}[H]
\centering
\includegraphics[width=0.8\textwidth]{images/your_image.png}
\caption{Your caption here}
\label{fig:your_label}
\end{figure}
```

### Modifying References
The bibliography is included in the main document. To add references:
1. Add new entries in the `thebibliography` environment
2. Follow IEEE format
3. Cite in text using `\cite{key}`

### Adjusting Section Depth
To change TOC depth, add before `\begin{document}`:
```latex
\setcounter{tocdepth}{3}  % Show up to subsections
\setcounter{secnumdepth}{3}
```

## Converting to Microsoft Word

### Method 1: Using Online Converters
1. Compile the LaTeX to PDF
2. Use online PDF to Word converters:
   - Smallpdf
   - ILovePDF
   - Adobe Acrobat online

### Method 2: Using Pandoc (Advanced)
```bash
pandoc FluoClean_AI_Report.tex -o FluoClean_AI_Report.docx
```
Note: Some formatting may need manual adjustment.

### Method 3: Copy-Paste
1. Open the compiled PDF
2. Copy content section by section
3. Paste into Word
4. Reformat as needed

## Final Checklist Before Submission

- [ ] All personal information placeholders replaced
- [ ] Institution logo added (or removed)
- [ ] Guide and HOD names verified
- [ ] Academic year updated
- [ ] References checked for accuracy
- [ ] Figures and tables properly labeled
- [ ] PDF compiles without errors
- [ ] Page numbers correct
- [ ] Table of contents complete
- [ ] All sections present
- [ ] Spelling and grammar checked
- [ ] Formatting consistent throughout

## Estimated Timeline

- Customization: 30-45 minutes
- First compilation: 1-2 minutes
- Review and adjustments: 1-2 hours
- Final compilation: 1 minute
- **Total: 2-3 hours**

## Need Help?

If you encounter issues:
1. Check the README.md for detailed information
2. Review the compile.log file for specific errors
3. Try alternative compilers (xelatex, lualatex)
4. Use Overleaf as a fallback option

## Success Indicators

You'll know everything is working when:
- PDF generates without errors
- All pages are present (35-40 pages)
- Table of contents is complete
- Figures and tables appear correctly
- Personal information shows your details
- Formatting looks professional

Good luck with your submission!
