# ECG Analyzer - Documentation

This directory contains the GitHub Pages website for ECG Analyzer.

## Pages

- **[Home](index.html)** - Main landing page
- **[Privacy Policy](privacy.html)** - Complete privacy policy for App Store compliance
- **[Support](support.html)** - FAQ and support information

## URLs

Once GitHub Pages is enabled for this repository:

- **Privacy Policy**: https://solomona8.github.io/cyclingECG/privacy.html
- **Support**: https://solomona8.github.io/cyclingECG/support.html

## Enabling GitHub Pages

1. Go to repository Settings
2. Navigate to Pages section
3. Under "Source", select branch: `claude/apple-watch-data-extraction-01GXvTS5e9r2psgrXhe73bva`
4. Select folder: `/docs`
5. Click Save
6. Wait 2-3 minutes for deployment

## Local Testing

To test these pages locally, you can use any static web server:

```bash
# Using Python
cd docs
python3 -m http.server 8000

# Using Node.js
npx http-server docs

# Using PHP
cd docs
php -S localhost:8000
```

Then visit http://localhost:8000 in your browser.
