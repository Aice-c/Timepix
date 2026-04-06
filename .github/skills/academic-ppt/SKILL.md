---
name: academic-ppt
description: "Generate professional academic PowerPoint presentations using python-pptx. Use when: creating research presentations, group meeting slides, thesis defense slides, conference talk slides, academic report slides. Produces clean, minimal, data-driven slides with proper typography."
argument-hint: "Describe the topic, data sources, and audience for the presentation"
---

# Academic PPT Generation

## When to Use
- Creating group meeting / lab meeting presentations
- Summarizing research findings into slide format
- Preparing thesis defense or conference talk slides
- Converting analysis reports into visual presentations

## Design Principles
1. **Clean & minimal**: No decorative elements, white/light background, consistent fonts
2. **Data-driven**: Embed charts/figures directly, use tables for numerical results
3. **Academic rigor**: Include proper units, significance levels, sample sizes
4. **Readable**: Large fonts (≥18pt body, ≥28pt titles), high contrast
5. **Structured**: Title → Background → Methods → Results → Conclusion → Next Steps

## Slide Layout Standards
- **Title slide**: Project name, presenter, date, affiliation
- **Content slides**: One key message per slide, bullet points ≤6 lines
- **Figure slides**: Full-width figure with brief caption
- **Table slides**: Clean tables with alternating row shading
- **Summary slides**: Key takeaways in bold numbered list

## Typography
- Title font: 28-36pt, bold, dark color (#1a1a2e or #2c3e50)
- Body font: 18-22pt, regular, dark gray (#333333)
- Caption/footnote: 12-14pt, italic, gray (#666666)
- Font family: Calibri or Arial (cross-platform safe)

## Color Palette (Academic)
- Primary: #2c3e50 (dark navy for titles)
- Accent: #2980b9 (blue for emphasis)
- Text: #333333 (dark gray body)
- Background: #FFFFFF (white)
- Table header: #2c3e50 with white text
- Table alt row: #f0f4f8 (light blue-gray)
- Positive result: #27ae60 (green)
- Negative/warning: #c0392b (red)

## Procedure
1. Read the analysis report / data sources to understand content
2. Plan slide structure (outline with ~8-15 slides)
3. Create a Python script using `python-pptx` that:
   - Sets slide dimensions to 16:9 (widescreen)
   - Applies consistent styling via helper functions
   - Embeds existing figure files (PNG) directly
   - Generates clean tables from data
   - Adds slide numbers
4. Run the script to generate the .pptx file
5. Verify the output file exists and report slide count

## Technical Notes
- Use `python-pptx` library (install via `pip install python-pptx`)
- Slide size: `prs.slide_width = Inches(13.333)`, `prs.slide_height = Inches(7.5)` for 16:9
- Always use `Pt()` for font sizes, `Inches()` for positioning
- Embed images with `slide.shapes.add_picture()`, scale to fit
- For tables: use `slide.shapes.add_table()` with proper column widths

## Reference Script Template
See [./scripts/ppt_template.py](./scripts/ppt_template.py) for reusable helper functions.
