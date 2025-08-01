# Changelog

All notable changes to the Support Ticket Summarizer project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.1.0] - 2025-01-31

### Changed - BREAKING
- **Telemetry Migration**: Complete migration from LangSmith to Langfuse Cloud due to CrewAI 0.80+ incompatibility
- **Session Management**: Implemented Option B intelligent session management (individual vs batch sessions)
- **Database Schema**: Complete database reset with enhanced collaboration metrics tracking

### Added
- **OpenInference Instrumentation**: Automatic tracing with OTLP exporter for Langfuse Cloud
- **Batch Session Management**: `create_batch_session()` method for grouping related ticket processing
- **Dynamic DeepEval Metrics**: Real-time calculation of hallucination, relevancy, and faithfulness scores
- **Enhanced Trace Context**: Comprehensive metadata including processing type and collaboration metrics
- **Custom Faithfulness Pipeline**: GPT-4o-based fact-checking with fallback mechanisms

### Fixed
- **CrewAI Compatibility**: Resolved callback system conflicts with CrewAI 0.80+ framework
- **DeepEval Integration**: Fixed hardcoded placeholder values, now showing authentic dynamic scores
- **Session Tracking**: Optimized Langfuse dashboard organization with logical session grouping
- **Database Constraints**: Enhanced collaboration metrics storage with proper error handling

### Technical Details
- **Migration Reason**: LangSmith callback system incompatible with new CrewAI memory architecture
- **Session Strategy**: Individual tickets = unique sessions, batch processing = shared session per batch
- **Quality Metrics**: Hallucination (1.000), Relevancy (1.000), Faithfulness (0.600) now calculated dynamically
- **Trace Capture**: Fully automatic via OpenInference - no manual callback management required

## [2.0.0] - 2024-12-XX

### Added - Major Architecture Overhaul
- **Collaborative Multi-Agent System**: Four specialized AI agents with authentic consensus building
- **Multi-Provider AI Support**: OpenAI, Anthropic, and Cohere integration with dynamic model assignment
- **Advanced Observability**: LangSmith integration with comprehensive tracing (later migrated to Langfuse)
- **Custom Quality Assessment**: GPT-4o-based faithfulness evaluation and DeepEval integration
- **Real-Time Dashboard**: Live agent monitoring with processing indicators
- **Database Analytics**: PostgreSQL with performance metrics and collaboration tracking
- **Dynamic Model Management**: Hot-swappable AI models per agent without system restart

### Changed
- **Processing Model**: From sequential to collaborative agent processing with real consensus building
- **Architecture**: Implemented authentic multi-agent collaboration with disagreement detection
- **Evaluation System**: Custom faithfulness scoring with multiple quality metrics
- **User Interface**: Complete Streamlit dashboard overhaul with real-time monitoring

## [1.0.0] - Initial Release

### Added
- Basic support ticket processing
- Single-agent architecture
- Simple classification and summarization
- Basic web interface
- SQLite database storage

---

## Migration Notes

### LangSmith → Langfuse Migration (v2.1.0)
This was a **critical breaking change** required due to framework incompatibility:

**Problem**: CrewAI 0.80+ introduced memory system changes that conflicted with LangSmith's callback architecture, causing:
- Failed trace captures during multi-agent collaboration
- Incomplete observability of consensus building processes
- Callback registration conflicts with CrewAI internals

**Solution**: Complete migration to Langfuse Cloud with OpenInference instrumentation:
- Automatic trace capture without callback conflicts
- Enhanced session management for better dashboard organization
- Improved compatibility with CrewAI's collaborative architecture

### DeepEval Metrics Evolution (v2.1.0)
Fixed evaluation system to provide authentic quality assessment:

**Before**: Hardcoded placeholder values that didn't reflect actual performance
**After**: Dynamic calculation of hallucination detection, relevancy scoring, and custom faithfulness evaluation

This change provides genuine insights into AI agent performance and output quality.