---
title: "journalGPT - AI-Powered Personal Journaling System"
excerpt: "An intelligent journaling application that leverages LLMs to generate personalized journal entries and provide automated summarization and insights, built with PostgreSQL, Docker, and local LLM inference."
collection: portfolio
date: 2025-10-02
permalink: /portfolio/journalgpt
---

## Overview

journalGPT is an AI-powered journaling system that combines the therapeutic benefits of journaling with the creativity of large language models. The application generates personalized, realistic journal entries based on user mood and thematic tags, while automatically creating summaries and insights to help users track their emotional journey over time.

**GitHub Repository**: [skapoor2024/journalGPT](https://github.com/skapoor2024/journalGPT)

## Key Features

###  AI-Generated Journal Entries
- **Mood-Based Generation**: Creates authentic journal entries that reflect the user's specified emotional state
- **Thematic Tags**: Incorporates specific themes (work, relationships, personal growth, etc.) into entries
- **Natural Language**: Generates first-person narratives that capture raw emotions and thoughts
- **Customizable Length**: Configurable word counts (default 400-500 words) for entry generation

###  Automatic Summarization
- **Event-Driven Architecture**: Triggers automatic summarization when journal entries are created
- **Key Insight Extraction**: Captures essential events and emotions from longer entries
- **Tone Preservation**: Maintains the original emotional context in condensed form
- **Multi-Tier Summaries**: Supports daily, weekly, and monthly aggregation patterns

###  Technical Architecture
- **PostgreSQL Database**: Robust relational database schema for users, entries, tags, and summaries
- **LLM Integration**: Local inference using Ollama with Gemma 3 model (1B parameter variant)
- **Docker Compose**: Containerized services for easy deployment and scalability
- **Event Handlers**: SQLAlchemy-based event system for automated workflows
- **Python Backend**: Modular service architecture with clear separation of concerns

## Technical Implementation

### System Architecture

The application follows a multi-service architecture:

1. **Database Layer** (`PostgreSQL`)
   - Users table for authentication and profiles
   - Journal entries with mood, tags, and timestamps
   - Tag management system for categorization
   - Summaries table with hierarchical aggregation support

2. **LLM Service** (`Ollama + Gemma 3`)
   - Local model inference for privacy and cost efficiency
   - Containerized deployment for isolation
   - Custom prompt engineering for journal and summary generation
   - Quantized model (QAT) for optimized performance

3. **Application Services**
   - `DataGenService`: Generates journal entries using LLM prompts
   - `SummarizationService`: Creates summaries with event-driven triggers
   - `JournalEngine`: Orchestrates entry creation and tag resolution

### Key Technologies

- **Backend**: Python 3.x with SQLAlchemy ORM
- **Database**: PostgreSQL with custom schemas and event triggers
- **LLM**: Ollama (Gemma 3:1B-IT-QAT)
- **Infrastructure**: Docker & Docker Compose
- **Dependency Management**: UV lock for reproducible builds

### Prompt Engineering

The system uses structured prompts with clear task definitions, requirements, and output specifications:

**Story Generation**:
```python
- Realistic first-person perspective
- Mood-appropriate tone and language
- Theme integration (e.g., work, relationships)
- Raw emotional expression
- 400-500 word length
```

**Summarization**:
```python
- 4-5 sentence condensation
- Key event and emotion capture
- Cohesive narrative structure
- Original tone preservation
```

## Development Timeline

- **August 2025**: Initial architecture and diagram planning
- **September 2025**: Database schema design and LLM service integration
- **October 2025**: Event handler implementation and prompt optimization
- **Current Status**: Active development with summarization features and data generation services

## Technical Highlights

### Database Schema Design
- Normalized relational structure with proper foreign key constraints
- Support for hierarchical tag systems
- Timestamp tracking for all entities
- Flexible summary types (daily, weekly, monthly)

### Event-Driven Summarization
- PostgreSQL triggers for automatic summary generation
- Asynchronous processing for performance optimization
- Configurable aggregation windows

### Containerization Strategy
- Service isolation with Docker Compose
- Health checks for database readiness
- Persistent volume management
- Environment-based configuration

## Future Enhancements

Based on the project roadmap:

- ✅ Sequence and architecture diagrams (Completed)
- ✅ LLM service for insight generation (Completed)
- ✅ Functional database and LLM integration (Completed)
- 🔄 Advanced insight analytics and trend detection
- 🔄 Multi-user support with authentication
- 🔄 Web interface for entry viewing and management
- 🔄 Export functionality (PDF, Markdown)


## Project Impact

journalGPT represents a novel application of generative AI for personal wellness and self-reflection. By combining the benefits of regular journaling with AI assistance, the system enables:

- **Privacy**: Local LLM inference keeps personal data secure
- **Insights**: Automated summarization reveals patterns over time
- **Accessibility**: Natural language interaction lowers barriers to journaling

---

