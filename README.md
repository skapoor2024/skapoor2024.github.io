# Shantanu Kapoor - AI/ML Engineer Portfolio

[![GitHub Pages](https://img.shields.io/badge/GitHub%20Pages-Live-green)](https://skapoor2024.github.io)
[![Built with Jekyll](https://img.shields.io/badge/Built%20with-Jekyll-red)](https://jekyllrb.com/)
[![Academic Template](https://img.shields.io/badge/Template-Academic%20Pages-blue)](https://github.com/academicpages/academicpages.github.io)

Professional portfolio website showcasing AI/ML engineering projects, research publications, and technical expertise.

## 🎯 About This Portfolio

This repository contains the source code for my professional portfolio website, built using Jekyll and the Academic Pages template. The site highlights my experience in:

- **Enterprise AI Automation**: LLM agents for SAP Financial Accounting
- **Computer Vision**: Medical image segmentation and satellite imagery processing  
- **Research Innovation**: Published work in multilingual speech processing
- **Production Systems**: Scalable ML pipelines and optimization

## 🚀 Key Projects Featured

### Enterprise AI
- **SAP FI Automation**: LLM-based agents using LangChain and FastAPI
- **Real-time Integration**: SAP OData APIs for financial data processing
- **Compliance Architecture**: Enterprise-grade audit trails and monitoring

### Computer Vision Research
- **Text-Guided Segmentation**: SwinUNETR + CLIP for medical imaging (Dice: 0.88)
- **Satellite Pipeline Optimization**: 80%+ processing time reduction
- **Materials Analysis**: Novel SAM enhancements for grain segmentation

### Machine Learning Innovation
- **Multilingual Speech Processing**: Novel similarity loss functions (ICASSP 2021)
- **Domain Adaptation**: Cross-device robustness in spectroscopic analysis
- **Performance Optimization**: HPC clusters and distributed computing

## 🛠 Technical Stack

**Frontend**: Jekyll, Liquid, HTML5, CSS3, JavaScript  
**Hosting**: GitHub Pages  
**Analytics**: Google Analytics (optional)  
**SEO**: Meta tags, structured data, sitemap optimization

## 📈 Features

- **Responsive Design**: Mobile-optimized professional presentation
- **SEO Optimized**: Search engine friendly structure and meta tags
- **Fast Loading**: Optimized assets and efficient Jekyll build
- **Professional Layout**: Clean, academic-style design suitable for recruiters
- **Interactive Elements**: Project showcases with technical details

## 📄 Content Structure

```
├── _pages/              # Main site pages
│   ├── about.md         # Professional summary and value proposition
│   ├── skills.md        # Technical skills and expertise
│   ├── experience.md    # Professional timeline and achievements
│   ├── contact.md       # Contact information and collaboration
│   └── cv.md           # Downloadable resume format
├── _portfolio/          # Project showcases
│   ├── sap-fi-automation.md
│   ├── medical-image-segmentation.md
│   ├── satellite-imagery-pipeline.md
│   └── ...
├── _publications/       # Research publications
│   ├── 2021-06-01-icassp-language-identification.md
│   ├── 2023-08-01-text-guided-segmentation.md
│   └── ...
└── files/              # Downloadable documents (PDFs, etc.)
```

## 🎯 Professional Highlights

- **Current Role**: AI Engineer at Hexaware Technologies
- **Education**: MS Computer Science (University of Florida), BTech CSE (IIT Mandi)
- **Publications**: ICASSP 2021, Under review at top-tier journals
- **Expertise**: Enterprise AI, Computer Vision, NLP, Production ML Systems
- **Location**: Gainesville, FL (Open to relocation)

## 📞 Contact Information

- **Email**: s.kapoor.ntk@outlook.com
- **LinkedIn**: [linkedin.com/in/skapoor13](https://linkedin.com/in/skapoor13)
- **GitHub**: [github.com/skapoor2024](https://github.com/skapoor2024)
- **Website**: [skapoor2024.github.io](https://skapoor2024.github.io)

## 🚀 Development & Deployment

### Running Locally

To preview changes locally before deployment:

1. Clone the repository and made updates as detailed above.
1. Make sure you have ruby-dev, bundler, and nodejs installed
    
    On most Linux distribution and [Windows Subsystem Linux](https://learn.microsoft.com/en-us/windows/wsl/about) the command is:
    ```bash
    sudo apt install ruby-dev ruby-bundler nodejs
    ```
    On MacOS the commands are:
    ```bash
    brew install ruby
    brew install node
    gem install bundler
    ```
1. Run `bundle install` to install ruby dependencies. If you get errors, delete Gemfile.lock and try again.
1. Run `jekyll serve -l -H localhost` to generate the HTML and serve it from `localhost:4000` the local server will automatically rebuild and refresh the pages on change.

If you are running on Linux it may be necessary to install some additional dependencies prior to being able to run locally: `sudo apt install build-essential gcc make`

# Maintenance

Bug reports and feature requests to the template should be [submitted via GitHub](https://github.com/academicpages/academicpages.github.io/issues/new/choose). For questions concerning how to style the template, please feel free to start a [new discussion on GitHub](https://github.com/academicpages/academicpages.github.io/discussions).

This repository was forked (then detached) by [Stuart Geiger](https://github.com/staeiou) from the [Minimal Mistakes Jekyll Theme](https://mmistakes.github.io/minimal-mistakes/), which is © 2016 Michael Rose and released under the MIT License (see LICENSE.md). It is currently being maintained by [Robert Zupko](https://github.com/rjzupkoii) and additional maintainers would be welcomed.

## Bugfixes and enhancements

If you have bugfixes and enhancements that you would like to submit as a pull request, you will need to [fork](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/fork-a-repo) this repository as opposed to using it as a template. This will also allow you to [synchronize your copy](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/syncing-a-fork) of template to your fork as well.

Unfortunately, one logistical issue with a template theme like Academic Pages that makes it a little tricky to get bug fixes and updates to the core theme. If you use this template and customize it, you will probably get merge conflicts if you attempt to synchronize. If you want to save your various .yml configuration files and markdown files, you can delete the repository and fork it again. Or you can manually patch.
