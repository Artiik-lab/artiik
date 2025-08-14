# ContextManager Documentation

This directory contains the complete documentation for ContextManager.

## 🚀 Quick Start

To serve the documentation locally:

```bash
# Option 1: Use the provided script
python3 ../serve_docs.py

# Option 2: Use Python's built-in server
python3 -m http.server 8080

# Option 3: Use any static file server
npx serve .
```

Then open your browser to: http://localhost:8080

## 📁 Files

- `index.html` - Main documentation page with Docsify
- `_sidebar.md` - Navigation sidebar
- `index.md` - Homepage content
- `getting_started.md` - Installation and basic usage
- `core_concepts.md` - Architecture and design concepts
- `api_reference.md` - Complete API documentation
- `examples.md` - Usage examples and patterns
- `advanced_usage.md` - Advanced features and customization
- `troubleshooting.md` - Common issues and solutions
- `architecture.md` - System design and performance
- `contributing.md` - Development guidelines

## 🎨 Features

- **Dark Mode**: Toggle between light and dark themes
- **Search**: Full-text search across all documentation
- **Syntax Highlighting**: Code blocks with proper highlighting
- **Copy Code**: One-click copying of code examples
- **Responsive Design**: Works on all devices
- **Professional Styling**: Modern, clean design

## 🔧 Customization

The documentation uses Docsify with custom CSS. To modify:

1. Edit `index.html` for configuration and styling
2. Edit `_sidebar.md` for navigation structure
3. Edit individual `.md` files for content

## 📦 Deployment

The documentation can be deployed to any static hosting service:

- GitHub Pages
- Netlify
- Vercel
- AWS S3
- Any static file server 